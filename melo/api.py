"""Text-to-Speech API module for MeloTTS.

Provides the :class:`TTS` class, which wraps the :class:`SynthesizerTrn` model
and exposes a high-level interface for loading models, splitting text into
sentences, and synthesising audio to a file or numpy array.
"""

from __future__ import annotations

# stdlib
import json
import os
import re

# third-party
import librosa
import numpy as np
import soundfile
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

# local
from . import commons
from . import utils
from .download_utils import load_or_download_config, load_or_download_model
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .models import SynthesizerTrn
from .split_utils import split_sentence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hidden dimension of the multilingual BERT encoder used for most languages.
BERT_DIM: int = 1024

# Hidden dimension of the Japanese BERT encoder (smaller model).
JA_BERT_DIM: int = 768


class TTS(nn.Module):
    """High-level Text-to-Speech interface built on top of SynthesizerTrn.

    Loads the model weights and configuration for a given language and
    exposes convenience methods for splitting text and synthesising audio.

    Attributes:
        model: The underlying :class:`SynthesizerTrn` synthesis network.
        symbol_to_id: Mapping from phoneme symbol strings to integer IDs.
        hps: Hyper-parameter namespace loaded from the config file.
        device: PyTorch device string on which inference is performed.
        language: Resolved language code used for text processing.
    """

    def __init__(
        self,
        language: str,
        device: str = 'auto',
        use_hf: bool = True,
        config_path: str | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Initialise the TTS model for the given language.

        Args:
            language: BCP-47-style language code, e.g. ``'EN'``, ``'ZH'``,
                ``'JP'``, ``'KR'``, ``'ES'``, ``'FR'``.
            device: PyTorch device string.  Use ``'auto'`` (default) to select
                CUDA > MPS > CPU automatically.
            use_hf: Whether to download missing assets from Hugging Face Hub.
            config_path: Optional path to a local ``config.json`` file.  When
                ``None`` the config is fetched via ``load_or_download_config``.
            ckpt_path: Optional path to a local model checkpoint.  When
                ``None`` the checkpoint is fetched via
                ``load_or_download_model``.
        """
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path =
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            n_vocab=len(symbols),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device

        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)

        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language  # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(
        segment_data_list: list[np.ndarray],
        sr: int,
        speed: float = 1.,
    ) -> np.ndarray:
        """Concatenate audio segments with short silence gaps between them.

        Args:
            segment_data_list: List of 1-D or multi-D numpy arrays containing
                raw audio samples for each synthesised sentence.
            sr: Sample rate in Hz used to compute the silence duration.
            speed: Playback speed factor.  Higher values shorten the gap.

        Returns:
            A single 1-D ``float32`` numpy array with all segments joined.
        """
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(
        text: str,
        language: str,
        quiet: bool = False,
    ) -> list[str]:
        """Split *text* into a list of sentence-level pieces.

        Args:
            text: Full input text to be split.
            language: Language code used by the underlying sentence splitter.
            quiet: When ``False`` (default) the split result is printed to
                stdout for debugging.

        Returns:
            A list of sentence strings ready for individual synthesis.
        """
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(
        self,
        text: str,
        speaker_id: int,
        output_path: str | None = None,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        speed: float = 1.0,
        pbar=None,
        format: str | None = None,
        position: int | None = None,
        quiet: bool = False,
    ) -> np.ndarray | None:
        """Synthesise *text* and write the result to *output_path*.

        Args:
            text: Input text to synthesise.
            speaker_id: Integer speaker ID from ``hps.data.spk2id``.
            output_path: Destination file path.  When ``None`` the audio array
                is returned instead of being written to disk.
            sdp_ratio: Stochastic duration predictor mixing ratio.
            noise_scale: Noise scale for the flow-based decoder.
            noise_scale_w: Noise scale for the duration predictor.
            speed: Speaking rate multiplier (1.0 = normal speed).
            pbar: Optional external progress-bar callable (e.g.
                ``gr.Progress().tqdm``).
            format: Audio format string passed to :func:`soundfile.write`
                (e.g. ``'wav'``).  Inferred from *output_path* when ``None``.
            position: ``tqdm`` position argument for nested progress bars.
            quiet: Suppress all stdout output when ``True``.

        Returns:
            A 1-D ``float32`` numpy array when *output_path* is ``None``;
            otherwise ``None`` (audio is written to file).
        """
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
