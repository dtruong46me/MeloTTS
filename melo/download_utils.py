"""Download and load utilities for MeloTTS model weights and configs.

This module provides helpers to:
- Resolve and download per-language model checkpoints and config files from
  either Hugging Face Hub or S3 (via ``cached_path``).
- Return loaded ``HParams`` configs and raw ``state_dict`` tensors ready for
  use by ``TTS`` and related classes.

Supported languages / locale prefixes:
    EN, EN_V2, EN_NEWEST, FR, JP, ES, ZH, KR
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
from typing import List

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
from cached_path import cached_path
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from . import utils


# ---------------------------------------------------------------------------
# Remote URL registries
# ---------------------------------------------------------------------------

# Mapping from language code to pre-trained checkpoint download URLs (S3)
DOWNLOAD_CKPT_URLS = {
    "EN": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth",
    "EN_V2": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth",
    "FR": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth",
    "JP": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth",
    "ES": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth",
    "ZH": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth",
    "KR": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth",
}

# Mapping from language code to config JSON download URLs (S3)
DOWNLOAD_CONFIG_URLS = {
    "EN": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/config.json",
    "EN_V2": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json",
    "FR": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/config.json",
    "JP": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/config.json",
    "ES": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/config.json",
    "ZH": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json",
    "KR": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/config.json",
}

# Mapping from filename to pretrained model download URLs (S3)
PRETRAINED_MODELS = {
    "G.pth": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth",
    "D.pth": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth",
    "DUR.pth": "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth",
}

# Mapping from language code to Hugging Face Hub repository IDs
LANG_TO_HF_REPO_ID = {
    "EN": "myshell-ai/MeloTTS-English",
    "EN_V2": "myshell-ai/MeloTTS-English-v2",
    "EN_NEWEST": "myshell-ai/MeloTTS-English-v3",
    "FR": "myshell-ai/MeloTTS-French",
    "JP": "myshell-ai/MeloTTS-Japanese",
    "ES": "myshell-ai/MeloTTS-Spanish",
    "ZH": "myshell-ai/MeloTTS-Chinese",
    "KR": "myshell-ai/MeloTTS-Korean",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_or_download_config(
    locale: str,
    use_hf: bool = True,
    config_path: str | None = None,
) -> utils.HParams:
    """Return an ``HParams`` config for the given locale, downloading if needed.

    If *config_path* is provided it is used directly. Otherwise the language
    code is extracted from *locale* and the config is fetched from Hugging Face
    Hub (when ``use_hf=True``) or from S3 via ``cached_path``.

    Args:
        locale: BCP-47 locale string or bare language code (e.g. ``"EN"``,
            ``"zh-CN"``). The language is derived by splitting on ``"-"`` and
            upper-casing the first part.
        use_hf: If ``True`` (default), download from Hugging Face Hub.
            If ``False``, download from the S3 URLs in ``DOWNLOAD_CONFIG_URLS``.
        config_path: Optional explicit path to a local config JSON file.
            When provided, *locale* and *use_hf* are ignored.

    Returns:
        An ``HParams`` instance parsed from the resolved config JSON file.
    """
    if config_path is None:
        language = locale.split("-")[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            config_path = hf_hub_download(
                repo_id=LANG_TO_HF_REPO_ID[language],
                filename="config.json",
            )
        else:
            assert language in DOWNLOAD_CONFIG_URLS
            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    return utils.get_hparams_from_file(config_path)


def load_or_download_model(
    locale: str,
    device: str,
    use_hf: bool = True,
    ckpt_path: str | None = None,
) -> dict:
    """Load a model checkpoint for the given locale, downloading if needed.

    If *ckpt_path* is provided it is used directly. Otherwise the language
    code is extracted from *locale* and the checkpoint is fetched from Hugging
    Face Hub (when ``use_hf=True``) or from S3 via ``cached_path``.

    Args:
        locale: BCP-47 locale string or bare language code (e.g. ``"EN"``,
            ``"zh-CN"``). The language is derived by splitting on ``"-"`` and
            upper-casing the first part.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``). The
            checkpoint is mapped to this device via ``map_location``.
        use_hf: If ``True`` (default), download from Hugging Face Hub.
            If ``False``, download from the S3 URLs in ``DOWNLOAD_CKPT_URLS``.
        ckpt_path: Optional explicit path to a local checkpoint ``.pth`` file.
            When provided, *locale* and *use_hf* are ignored.

    Returns:
        The raw checkpoint dictionary (as returned by ``torch.load``), typically
        containing ``"model"``, ``"iteration"``, and ``"optimizer"`` keys.
    """
    if ckpt_path is None:
        language = locale.split("-")[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            ckpt_path = hf_hub_download(
                repo_id=LANG_TO_HF_REPO_ID[language],
                filename="checkpoint.pth",
            )
        else:
            assert language in DOWNLOAD_CKPT_URLS
            ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])
    return torch.load(ckpt_path, map_location=device)


def load_pretrain_model() -> List[str]:
    """Download all pretrained model files (G, D, DUR) and return their local paths.

    Files are fetched via ``cached_path`` from the URLs registered in
    ``PRETRAINED_MODELS``. Already-cached files are not re-downloaded.

    Returns:
        A list of local file-system paths to the downloaded ``G.pth``,
        ``D.pth``, and ``DUR.pth`` checkpoint files, in iteration order of
        ``PRETRAINED_MODELS``.
    """
    return [cached_path(url) for url in PRETRAINED_MODELS.values()]
