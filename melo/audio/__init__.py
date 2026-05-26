"""Audio / DSP utilities for MeloTTS.

Sub-package containing signal processing helpers:

* :mod:`melo.audio.mel_processing` — STFT, mel-spectrogram computation, and
  related utilities used during both training and inference.

Key functions are re-exported at this level::

    from melo.audio import spectrogram_torch, mel_spectrogram_torch
"""

from __future__ import annotations

from .mel_processing import (  # noqa: F401
    dynamic_range_compression_torch,
    dynamic_range_decompression_torch,
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectral_de_normalize_torch,
    spectral_normalize_torch,
    spectrogram_torch,
    spectrogram_torch_conv,
)
