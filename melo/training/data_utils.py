"""Dataset utilities for multi-speaker text-audio training in MeloTTS.

This module provides:

* :class:`TextAudioSpeakerLoader` — a ``torch.utils.data.Dataset`` that loads
  (audio, speaker-ID, text) triplets, normalises them, and computes
  spectrograms on the fly.
* :class:`TextAudioSpeakerCollate` — a collate function that zero-pads a batch
  of variable-length samples into fixed-size tensors.
* :class:`DistributedBucketSampler` — a distributed sampler that groups
  samples by audio length to reduce within-batch padding.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from loguru import logger
from tqdm import tqdm

from melo.nn import commons
from melo.audio.mel_processing import spectrogram_torch, mel_spectrogram_torch
from melo.utils.core import load_filepaths_and_text
from melo.utils.core import load_wav_to_torch_librosa as load_wav_to_torch
from melo.text import cleaned_text_to_sequence, get_bert


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """Dataset that loads (audio, speaker-ID, text) triplets for training.

    Pipeline per sample:
    1. Load the audio waveform and resample to ``hparams.sampling_rate``.
    2. Compute (or load a cached) linear/mel spectrogram.
    3. Convert the pre-cleaned phoneme string to integer ID sequences and
       optionally interleave blank tokens.
    4. Load or compute BERT embeddings and assign them to the correct slot
       (Chinese BERT vs. Japanese/other BERT) based on the language code.

    Attributes:
        audiopaths_sid_text: List of ``[audiopath, spk, language, text,
            phones, tone, word2ph]`` rows after filtering.
        lengths: List of approximate spectrogram frame counts, used by
            :class:`DistributedBucketSampler` for bucketing.
        hparams: Hyperparameter object forwarded from the constructor.
        spk_map: Mapping from speaker name to integer speaker ID.
        disable_bert: When ``True``, BERT tensors are replaced with zeros.
    """

    def __init__(self, audiopaths_sid_text: str, hparams) -> None:
        """Initialise the dataset and filter samples by phoneme length.

        Args:
            audiopaths_sid_text: Path to the file listing audio paths together
                with speaker ID, language, text and phoneme information.
            hparams: Hyperparameter namespace / object exposing at minimum:
                ``max_wav_value``, ``sampling_rate``, ``filter_length``,
                ``hop_length``, ``win_length``, ``spk2id``, ``add_blank``.
        """
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.disable_bert = getattr(hparams, "disable_bert", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self) -> None:
        """Filter samples by phoneme length and compute spectrogram lengths.

        Samples whose phoneme count falls outside
        ``[min_text_len, max_text_len]`` are discarded.  For the remaining
        samples, an approximate spectrogram frame count is stored in
        ``self.lengths`` (used by :class:`DistributedBucketSampler`).

        The approximation follows::

            spec_length ≈ file_size_bytes // (2 * hop_length)

        because WAV PCM is 1 channel × 2 bytes per sample.
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(
            self.audiopaths_sid_text
        ):
            try:
                _id, spk, language, text, phones, tone, word2ph = item
            except Exception:
                print(item)
                raise
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}')
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(
        self,
        audiopath_sid_text: list,
    ) -> tuple:
        """Load and process a single (audio, text, speaker) sample.

        Args:
            audiopath_sid_text: A list of 7 elements
                ``[audiopath, sid, language, text, phones, tone, word2ph]``.

        Returns:
            An 8-tuple ``(phones, spec, wav, sid, tone, language, bert,
            ja_bert)`` of tensors ready for collation.
        """
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text

        bert, ja_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )

        spec, wav = self.get_audio(audiopath)
        sid = int(getattr(self.spk_map, sid, '0'))
        sid = torch.LongTensor([sid])
        return (phones, spec, wav, sid, tone, language, bert, ja_bert)

    def get_audio(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an audio file and compute its spectrogram.

        The spectrogram is cached as a ``.spec.pt`` (or ``.mel.pt``) file
        alongside the source ``.wav`` file.  If a cached file is present it is
        loaded directly; otherwise the spectrogram is computed, saved, and
        returned.

        Args:
            filename: Absolute path to the ``.wav`` audio file.

        Returns:
            A 2-tuple ``(spec, audio_norm)`` where:

            * **spec** – spectrogram tensor of shape
              ``(n_fft // 2 + 1, T)`` or ``(n_mel_channels, T)``.
            * **audio_norm** – normalised waveform tensor of shape
              ``(1, num_samples)``.

        Raises:
            ValueError: If the file's sampling rate does not match
                ``self.sampling_rate``.
        """
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        # NOTE: normalize has been achieved by torchaudio
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
            assert False
        except Exception:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(
        self,
        text: str,
        word2ph: list[int],
        phone: list[str],
        tone: list[int],
        language_str: str,
        wav_path: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert phoneme sequences to tensors and retrieve BERT features.

        Optionally inserts blank tokens between phones (when ``add_blank`` is
        set).  BERT embeddings are loaded from a cached ``.bert.pt`` file if
        available; otherwise they are computed and cached.

        Language routing for BERT tensors:

        * ``"ZH"`` → ``bert`` slot (1024-dim); ``ja_bert`` is zeros (768-dim).
        * All other supported languages → ``ja_bert`` slot (768-dim); ``bert``
          is zeros (1024-dim).

        Args:
            text: Normalised text string (used to recompute BERT if cache miss).
            word2ph: Word-to-phoneme count mapping.
            phone: List of phoneme symbol strings.
            tone: List of raw tone integers.
            language_str: Language code string (e.g. ``"ZH"``, ``"JP"``).
            wav_path: Path to the corresponding ``.wav`` file; used to derive
                the BERT cache path.

        Returns:
            A 5-tuple ``(bert, ja_bert, phone, tone, language)`` of
            ``torch.Tensor`` objects.
        """
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert = torch.load(bert_path)
            assert bert.shape[-1] == len(phone)
        except Exception as e:
            print(e, wav_path, bert_path, bert.shape, len(phone))
            bert = get_bert(text, word2ph, language_str)
            torch.save(bert, bert_path)
            assert bert.shape[-1] == len(phone), phone

        if self.disable_bert:
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(768, len(phone))
        else:
            if language_str in ["ZH"]:
                bert = bert
                ja_bert = torch.zeros(768, len(phone))
            elif language_str in ["JP", "EN", "ZH_MIX_EN", "KR", 'SP', 'ES', 'FR', 'DE', 'RU']:
                ja_bert = bert
                bert = torch.zeros(1024, len(phone))
            else:
                raise
                bert = torch.zeros(1024, len(phone))  # NOTE: unreachable code - kept for reference
                ja_bert = torch.zeros(768, len(phone))  # NOTE: unreachable code - kept for reference
        assert bert.shape[-1] == len(phone)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, phone, tone, language

    def get_sid(self, sid: str) -> torch.Tensor:
        """Convert a speaker ID string to a LongTensor.

        Args:
            sid: Speaker ID as a string representing an integer.

        Returns:
            A 1-element ``torch.LongTensor`` containing the speaker ID.
        """
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index: int) -> tuple:
        """Return the processed sample at the given index.

        Args:
            index: Index into ``self.audiopaths_sid_text``.

        Returns:
            The 8-tuple produced by :meth:`get_audio_text_speaker_pair`.
        """
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of (audio, text, speaker) triplets after filtering.
        """
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets into fixed-size batch tensors.

    Samples within a batch are sorted in decreasing spectrogram length order
    before padding so that packed-sequence operations (if used downstream)
    work correctly.

    Attributes:
        return_ids: Whether to return the sorted index mapping alongside the
            padded tensors.  Currently unused in the ``__call__`` return value.
    """

    def __init__(self, return_ids: bool = False) -> None:
        """Initialise the collate helper.

        Args:
            return_ids: If ``True``, the original sort indices are preserved
                (currently not surfaced in the return tuple).
        """
        self.return_ids = return_ids

    def __call__(self, batch: list[tuple]) -> tuple:
        """Collate a list of samples into a padded batch.

        Samples are sorted by spectrogram length (descending) before padding.
        All sequence tensors are right-zero-padded to the maximum length
        within the batch.

        Args:
            batch: List of 8-tuples
                ``(phones, spec, wav, sid, tone, language, bert, ja_bert)``
                as returned by :meth:`TextAudioSpeakerLoader.__getitem__`.

        Returns:
            An 11-tuple of tensors:
            ``(text_padded, text_lengths, spec_padded, spec_lengths,
            wav_padded, wav_lengths, sid, tone_padded, language_padded,
            bert_padded, ja_bert_padded)``.
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 768, max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        ja_bert_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            ja_bert = row[7]
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            bert_padded,
            ja_bert_padded,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """Distributed sampler that groups samples into length buckets.

    Maintains similar spectrogram lengths within each batch to reduce padding
    waste.  Bucket boundaries are specified as a sorted list of frame counts;
    any sample whose length falls outside the range ``(boundaries[0],
    boundaries[-1]]`` is silently discarded.

    Example::

        boundaries = [b1, b2, b3]
        # bucket 0: b1 < length <= b2
        # bucket 1: b2 < length <= b3

    Attributes:
        lengths: List of approximate spectrogram frame counts, one per sample.
        batch_size: Number of samples per batch.
        boundaries: Sorted list of length bucket boundaries.
        buckets: List of index lists, one per active bucket.
        num_samples_per_bucket: Padded sample count per bucket (divisible by
            ``num_replicas * batch_size``).
        total_size: Sum of ``num_samples_per_bucket``.
        num_samples: ``total_size // num_replicas`` — samples per replica.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        boundaries: list[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        """Initialise the sampler and create length buckets.

        Args:
            dataset: The dataset to sample from; must expose a ``lengths``
                attribute (list of approximate spectrogram frame counts).
            batch_size: Number of samples per batch.
            boundaries: Sorted list of frame-count boundaries that define the
                buckets.  At least 2 values are required.
            num_replicas: Number of distributed processes.  Defaults to the
                value inferred by :class:`torch.utils.data.distributed.DistributedSampler`.
            rank: Rank of the current process.  Defaults to the value inferred
                by the parent sampler.
            shuffle: Whether to shuffle indices within each bucket and the
                final batch order each epoch.
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        print('buckets:', self.num_samples_per_bucket)

    def _create_buckets(self) -> tuple[list[list[int]], list[int]]:
        """Partition dataset indices into length buckets.

        Empty buckets (after the first) are removed along with the
        corresponding boundary value.  For each surviving bucket the sample
        count is rounded up to the nearest multiple of
        ``num_replicas * batch_size`` so that every replica receives the same
        number of complete batches.

        Returns:
            A 2-tuple ``(buckets, num_samples_per_bucket)`` where:

            * **buckets** – list of index lists, one per active bucket.
            * **num_samples_per_bucket** – padded sample count per bucket.
        """
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        """Yield batches of dataset indices, grouped by length bucket.

        Each call uses ``self.epoch`` as the random seed to ensure deterministic
        yet different shuffling across epochs.

        Yields:
            Lists of dataset indices, each of length ``self.batch_size``.
        """
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x: int, lo: int = 0, hi: Optional[int] = None) -> int:
        """Find the bucket index for a sample of length ``x``.

        Uses a recursive binary search over ``self.boundaries``.

        Args:
            x: The length value to locate.
            lo: Lower boundary index (inclusive).  Defaults to ``0``.
            hi: Upper boundary index (exclusive).  Defaults to
                ``len(self.boundaries) - 1``.

        Returns:
            The 0-based bucket index if ``x`` falls within a valid bucket,
            otherwise ``-1``.
        """
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self) -> int:
        """Return the number of batches this sampler will yield.

        Returns:
            ``self.num_samples // self.batch_size``.
        """
        return self.num_samples // self.batch_size
