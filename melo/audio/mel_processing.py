"""Mel-spectrogram processing utilities for audio feature extraction.

This module provides PyTorch-based functions for computing spectrograms and
mel-spectrograms used in TTS (Text-to-Speech) pipelines. It includes dynamic
range compression/decompression helpers and both a standard STFT approach and
an equivalent convolution-based STFT (ConvSTFT) implementation.
"""

from __future__ import annotations

import torch
import torch.utils.data
import librosa
from librosa.filters import mel as librosa_mel_fn

# Maximum WAV amplitude for 16-bit PCM audio (2^15)
MAX_WAV_VALUE = 32768.0

# Module-level caches keyed by (window_size, dtype, device) or (fmax, dtype, device)
# to avoid re-creating tensors on every call.
mel_basis: dict[str, torch.Tensor] = {}
hann_window: dict[str, torch.Tensor] = {}


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """Apply dynamic range compression using a logarithmic scale.

    Args:
        x: Input tensor (typically magnitude spectrogram values).
        C: Compression factor applied as a multiplier before taking the log.
        clip_val: Minimum value to clamp ``x`` to before applying the log,
            preventing log(0) instability.

    Returns:
        Log-compressed tensor of the same shape as ``x``.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(
    x: torch.Tensor,
    C: float = 1,
) -> torch.Tensor:
    """Reverse dynamic range compression (inverse of compression_torch).

    Args:
        x: Log-compressed tensor to decompress.
        C: Compression factor used during the original compression step.

    Returns:
        Decompressed tensor of the same shape as ``x``.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Normalize spectrogram magnitudes via dynamic range compression.

    Args:
        magnitudes: Raw magnitude spectrogram tensor.

    Returns:
        Log-compressed (normalized) spectrogram tensor.
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Denormalize spectrogram magnitudes via dynamic range decompression.

    Args:
        magnitudes: Log-compressed spectrogram tensor.

    Returns:
        Decompressed magnitude spectrogram tensor.
    """
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Compute a magnitude spectrogram using torch.stft.

    Uses a cached Hann window (keyed by window size, dtype, and device) to
    avoid redundant tensor allocations across repeated calls.

    Args:
        y: Input waveform tensor of shape ``(batch, time)`` or ``(time,)``.
        n_fft: FFT size (number of frequency bins before onesided reduction).
        sampling_rate: Audio sampling rate in Hz (used for informational
            purposes; does not affect computation directly).
        hop_size: Number of samples between successive STFT frames.
        win_size: Length of the analysis window in samples.
        center: If ``True``, pad the signal so that frame ``t`` is centered at
            ``t * hop_size``. Must be ``False`` for the conv variant.

    Returns:
        Magnitude spectrogram tensor of shape
        ``(batch, n_fft // 2 + 1, frames)``.
    """
    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spectrogram_torch_conv(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Compute a magnitude spectrogram using a convolution-based STFT (ConvSTFT).

    Functionally equivalent to :func:`spectrogram_torch` but implements STFT
    as a 1-D convolution over pre-computed Fourier basis filters. Includes an
    assertion that verifies numerical equivalence with ``torch.stft``.

    Args:
        y: Input waveform tensor of shape ``(batch, time)`` or ``(time,)``.
        n_fft: FFT size (number of frequency bins before onesided reduction).
        sampling_rate: Audio sampling rate in Hz (not used in computation).
        hop_size: Number of samples between successive STFT frames.
        win_size: Length of the analysis window in samples.
        center: Must be ``False``; centering is not supported in this variant
            (see the commented-out block below).

    Returns:
        Magnitude spectrogram tensor of shape
        ``(batch, n_fft // 2 + 1, frames)``.
    """
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')

    # ******************** original ************************#
    # y = y.squeeze(1)
    # spec1 = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
    #                   center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft)))
    forward_basis = fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    forward_basis = forward_basis * torch.as_tensor(librosa.util.pad_center(torch.hann_window(win_size), size=n_fft)).float()

    import torch.nn.functional as F

    # if center:
    #     signal = F.pad(y[:, None, None, :], (n_fft // 2, n_fft // 2, 0, 0), mode = 'reflect').squeeze(1)
    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride = hop_size)
    spec2 = torch.stack([forward_transform_squared[:, :freq_cutoff, :], forward_transform_squared[:, freq_cutoff:, :]], dim = -1)


    # ******************** Verification ************************#
    spec1 = torch.stft(y.squeeze(1), n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    assert torch.allclose(spec1, spec2, atol=1e-4)

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Convert a linear-scale spectrogram to a mel-scale spectrogram.

    Uses a cached mel filterbank (keyed by ``fmax``, dtype, and device) to
    avoid redundant computations across repeated calls.

    Args:
        spec: Linear-scale magnitude spectrogram of shape
            ``(batch, n_fft // 2 + 1, frames)``.
        n_fft: FFT size used when computing ``spec``.
        num_mels: Number of mel filterbank channels.
        sampling_rate: Audio sampling rate in Hz.
        fmin: Minimum frequency for the mel filterbank in Hz.
        fmax: Maximum frequency for the mel filterbank in Hz.

    Returns:
        Log-compressed mel spectrogram of shape ``(batch, num_mels, frames)``.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> torch.Tensor:
    """Compute a log-compressed mel spectrogram directly from a waveform.

    Combines STFT computation and mel filterbank application in a single call.
    Uses module-level caches for both the mel filterbank and the Hann window.

    Args:
        y: Input waveform tensor of shape ``(batch, time)`` or ``(time,)``.
        n_fft: FFT size (number of frequency bins before onesided reduction).
        num_mels: Number of mel filterbank channels.
        sampling_rate: Audio sampling rate in Hz.
        hop_size: Number of samples between successive STFT frames.
        win_size: Length of the analysis window in samples.
        fmin: Minimum frequency for the mel filterbank in Hz.
        fmax: Maximum frequency for the mel filterbank in Hz.
        center: If ``True``, pad the signal so that frame ``t`` is centered at
            ``t * hop_size``.

    Returns:
        Log-compressed mel spectrogram of shape ``(batch, num_mels, frames)``.
    """
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
