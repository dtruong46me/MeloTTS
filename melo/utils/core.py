"""Utility functions for MeloTTS training and inference.

This module provides helpers for:
- Text preprocessing and BERT feature extraction (get_text_for_tts_infer)
- Checkpoint loading/saving (load_checkpoint, save_checkpoint)
- TensorBoard summarization (summarize)
- Audio loading in multiple backends (load_wav_to_torch, load_wav_to_torch_new, load_wav_to_torch_librosa)
- Hyperparameter management via HParams and associated loaders
- Logging utilities (get_logger)
- Spectrogram/alignment visualization (plot_spectrogram_to_numpy, plot_alignment_to_numpy)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import glob
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torchaudio
import librosa
from scipy.io.wavfile import read

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from melo.nn import commons
from melo.text import cleaned_text_to_sequence
from melo.text import get_bert
from melo.text.cleaner import clean_text
from melo.utils.config import ConfigSchema


# Flag to track whether matplotlib has been initialized with the Agg backend
MATPLOTLIB_FLAG: bool = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text / feature helpers
# ---------------------------------------------------------------------------


def get_text_for_tts_infer(
    text: str,
    language_str: str,
    hps: ConfigSchema,
    device: str,
    symbol_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare text inputs for TTS inference.

    Cleans raw text, converts it to phoneme sequences, and computes BERT
    embeddings. Returns tensors ready to be fed into the TTS model.

    Args:
        text: Raw input text string.
        language_str: Language identifier (e.g. ``"ZH"``, ``"EN"``, ``"JP"``).
        hps: Hyperparameter object containing at least ``hps.data`` settings.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
        symbol_to_id: Optional mapping from phoneme symbol to integer id.

    Returns:
        A tuple of ``(bert, ja_bert, phone, tone, language)`` tensors where:
        - ``bert``     — BERT embedding for Chinese (shape ``[1024, T]``).
        - ``ja_bert``  — BERT embedding for Japanese/other languages (shape ``[768, T]``).
        - ``phone``    — LongTensor of phoneme ids (shape ``[T]``).
        - ``tone``     — LongTensor of tone ids (shape ``[T]``).
        - ``language`` — LongTensor of language ids (shape ``[T]``).
    """
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(
            norm_text=norm_text,
            word2ph=word2ph,
            language=language_str,
            device=device,
        )
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert  # noqa: no-op, kept for symmetry
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", "KR", "SP", "ES", "FR", "DE", "RU"]:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skip_optimizer: bool = False,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], float, int]:
    """Load a training checkpoint from disk into a model (and optionally optimizer).

    Performs a lenient load — missing keys are filled with zeros and a warning
    is logged; extra keys are silently ignored (``strict=False``).

    Args:
        checkpoint_path: Absolute path to the ``.pth`` checkpoint file.
        model: The PyTorch model to load weights into.
        optimizer: Optional optimizer whose state will be restored.
        skip_optimizer: If ``True``, skip loading the optimizer state even when
            ``optimizer`` is not ``None``.

    Returns:
        A tuple ``(model, optimizer, learning_rate, iteration)`` where:
        - ``model``         — Model with loaded weights.
        - ``optimizer``     — Optimizer with (possibly) loaded state, or the
                              original value if skipped.
        - ``learning_rate`` — Learning rate stored in the checkpoint.
        - ``iteration``     — Training iteration stored in the checkpoint.
    """
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 0.0)
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception as e:
            print(e)
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warn(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            else:
                logger.error(f"{k} is not in the checkpoint")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )

    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: str,
) -> None:
    """Serialize model and optimizer state to a checkpoint file.

    Args:
        model: The PyTorch model to save. Handles ``DataParallel`` wrappers
            via ``.module``.
        optimizer: Optimizer whose ``state_dict`` will be saved.
        learning_rate: Current learning rate to embed in the checkpoint.
        iteration: Current training iteration to embed in the checkpoint.
        checkpoint_path: Destination file path for the ``.pth`` checkpoint.
    """
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------------


def summarize(
    writer: Any,
    global_step: int,
    scalars: Dict = {},
    histograms: Dict = {},
    images: Dict = {},
    audios: Dict = {},
    audio_sampling_rate: int = 22050,
) -> None:
    """Write scalars, histograms, images, and audio clips to a TensorBoard writer.

    Args:
        writer: A ``torch.utils.tensorboard.SummaryWriter`` instance.
        global_step: The current global training step (x-axis value).
        scalars: Mapping of tag → scalar value.
        histograms: Mapping of tag → tensor/array for histogram logging.
        images: Mapping of tag → HWC image array.
        audios: Mapping of tag → audio waveform array.
        audio_sampling_rate: Sampling rate used when adding audio summaries.
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> str:
    """Return the path of the latest checkpoint matching *regex* in *dir_path*.

    Files are sorted by extracting the numeric portion of the filename, so
    ``G_10000.pth`` sorts after ``G_9000.pth`` regardless of mtime.

    Args:
        dir_path: Directory to search for checkpoint files.
        regex: Glob pattern used to filter files (default ``"G_*.pth"``).

    Returns:
        Absolute path of the checkpoint with the highest numeric index.
    """
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """Render a spectrogram matrix as an RGB image array (HWC, uint8).

    Initializes matplotlib with the non-interactive ``Agg`` backend on the
    first call (controlled by the module-level ``MATPLOTLIB_FLAG``).

    Args:
        spectrogram: 2-D array of shape ``[channels, frames]``.

    Returns:
        RGB image as a NumPy array of shape ``[H, W, 3]`` with dtype ``uint8``.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(
    alignment: np.ndarray,
    info: Optional[str] = None,
) -> np.ndarray:
    """Render an alignment matrix as an RGB image array (HWC, uint8).

    Initializes matplotlib with the non-interactive ``Agg`` backend on the
    first call (controlled by the module-level ``MATPLOTLIB_FLAG``).

    Args:
        alignment: 2-D array of shape ``[decoder_T, encoder_T]``.
        info: Optional extra annotation appended below the x-axis label.

    Returns:
        RGB image as a NumPy array of shape ``[H, W, 3]`` with dtype ``uint8``.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


# ---------------------------------------------------------------------------
# Audio loaders
# ---------------------------------------------------------------------------


def load_wav_to_torch(full_path: str) -> Tuple[torch.Tensor, int]:
    """Load a WAV file using ``scipy.io.wavfile.read`` and return a FloatTensor.

    Args:
        full_path: Path to the WAV file.

    Returns:
        A tuple ``(audio_tensor, sampling_rate)`` where ``audio_tensor`` is a
        1-D ``torch.FloatTensor``.
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_torch_new(full_path: str) -> Tuple[torch.Tensor, int]:
    """Load a WAV file using ``torchaudio`` and return a mono FloatTensor.

    Channels are averaged to produce a single-channel (mono) waveform.

    Args:
        full_path: Path to the audio file (any format supported by torchaudio).

    Returns:
        A tuple ``(audio_norm, sampling_rate)`` where ``audio_norm`` is a
        1-D normalized ``torch.Tensor``.
    """
    audio_norm, sampling_rate = torchaudio.load(
        full_path,
        frame_offset=0,
        num_frames=-1,
        normalize=True,
        channels_first=True,
    )
    audio_norm = audio_norm.mean(dim=0)
    return audio_norm, sampling_rate


def load_wav_to_torch_librosa(full_path: str, sr: int) -> Tuple[torch.Tensor, int]:
    """Load a WAV file using ``librosa`` and return a mono FloatTensor.

    Args:
        full_path: Path to the audio file.
        sr: Target sampling rate. ``librosa`` will resample if necessary.

    Returns:
        A tuple ``(audio_tensor, sampling_rate)`` where ``audio_tensor`` is a
        1-D ``torch.FloatTensor``.
    """
    audio_norm, sampling_rate = librosa.load(full_path, sr=sr, mono=True)
    return torch.FloatTensor(audio_norm.astype(np.float32)), sampling_rate


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_filepaths_and_text(
    filename: str,
    split: str = "|",
) -> List[List[str]]:
    """Read a pipe-delimited (or custom delimiter) file into a list of rows.

    Each line is stripped of whitespace and split by *split*, yielding a list
    of string tokens per line.

    Args:
        filename: Path to the metadata text file.
        split: Column delimiter character (default ``"|"``).

    Returns:
        A list of rows, where each row is a list of string tokens.
    """
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


# ---------------------------------------------------------------------------
# Hyperparameter management
# ---------------------------------------------------------------------------


def get_hparams(init: bool = True) -> ConfigSchema:
    """Parse command-line arguments and build a hyperparameter object.

    Reads a JSON config file specified via ``--config``, copies it to the
    model directory, and attaches extra CLI arguments (pretrain paths, port,
    etc.) as attributes on the returned ``ConfigSchema`` instance.

    Args:
        init: If ``True``, copy the source config to ``logs/<model>/config.json``.
              If ``False``, read the config directly from the model directory.

    Returns:
        A ``ConfigSchema`` instance populated from the JSON config and CLI args,
        with ``model_dir``, ``pretrain_G``, ``pretrain_D``, ``pretrain_dur``,
        and ``port`` attributes attached.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("--pretrain_G", type=str, default=None, help="pretrain model")
    parser.add_argument("--pretrain_D", type=str, default=None, help="pretrain model D")
    parser.add_argument("--pretrain_dur", type=str, default=None, help="pretrain model duration")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = ConfigSchema(**config)
    hparams.model_dir = model_dir
    hparams.pretrain_G = args.pretrain_G
    hparams.pretrain_D = args.pretrain_D
    hparams.pretrain_dur = args.pretrain_dur
    hparams.port = args.port
    return hparams


def clean_checkpoints(
    path_to_models: str = "logs/44k/",
    n_ckpts_to_keep: int = 2,
    sort_by_time: bool = True,
) -> None:
    """Free up disk space by deleting old checkpoints.

    Keeps the *n_ckpts_to_keep* most recent (or highest-numbered) ``G_*.pth``
    and ``D_*.pth`` files, and never deletes ``G_0.pth`` or ``D_0.pth``.

    Args:
        path_to_models: Directory containing the checkpoint files.
        n_ckpts_to_keep: Number of checkpoints to retain per model type
            (``G`` and ``D``), excluding the ``_0`` baseline checkpoint.
        sort_by_time: If ``True``, sort (and delete oldest) by modification
            time. If ``False``, sort (and delete lowest-numbered) lexicographically
            by the numeric suffix in the filename.
    """
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f: str) -> int:
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f: str) -> float:
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x: str) -> List[str]:
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn: str) -> None:
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x: str) -> list:
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_hparams_from_dir(model_dir: str) -> ConfigSchema:
    """Load hyperparameters from a ``config.json`` file inside *model_dir*.

    Args:
        model_dir: Path to the model directory that contains ``config.json``.

    Returns:
        A ``ConfigSchema`` instance with ``model_dir`` set to the given directory.
    """
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = ConfigSchema(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path: str) -> ConfigSchema:
    """Load hyperparameters directly from a JSON config file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A ``ConfigSchema`` instance populated from the JSON file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = ConfigSchema(**config)
    return hparams

import subprocess

# ---------------------------------------------------------------------------
# Git / logging helpers
# ---------------------------------------------------------------------------


def check_git_hash(model_dir: str) -> None:
    """Compare the current git commit hash with the one saved in *model_dir*."""
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir: str, filename: str = "train.log") -> logging.Logger:
    """Create (or reconfigure) the module-level logger to also write to a file."""
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger
