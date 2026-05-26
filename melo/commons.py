"""Common utility functions for MeloTTS.

This module provides a collection of shared helper functions used across the
MeloTTS codebase, including weight initialization, padding utilities, signal
generation, sequence masking, and gradient clipping.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Union

import torch
from torch.nn import functional as F


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize convolutional layer weights with a normal distribution.

    Args:
        m: The module whose weights will be initialized.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute the padding size to maintain the same output length after convolution.

    Args:
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor of the convolution.

    Returns:
        The integer padding value.
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert a nested list of pad shapes into a flat list suitable for F.pad.

    Reverses the order of the dimensions and flattens the nested structure,
    as required by PyTorch's F.pad which expects padding in reverse dimension order.

    Args:
        pad_shape: A list of [pad_left, pad_right] pairs, one per dimension,
            ordered from outermost to innermost dimension.

    Returns:
        A flat list of padding values in the format expected by F.pad.
    """
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def intersperse(lst: list, item: object) -> list:
    """Insert an item between every element of a list.

    Args:
        lst: The original list of items.
        item: The item to intersperse between elements.

    Returns:
        A new list with ``item`` placed between every element of ``lst``,
        as well as at the beginning and end.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL divergence KL(P || Q) for two diagonal Gaussian distributions.

    Args:
        m_p: Mean of distribution P.
        logs_p: Log standard deviation of distribution P.
        m_q: Mean of distribution Q.
        logs_q: Log standard deviation of distribution Q.

    Returns:
        Element-wise KL divergence tensor of the same shape as the inputs.
    """
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape: Union[torch.Size, List[int]]) -> torch.Tensor:
    """Sample from the Gumbel distribution, protected from overflows.

    Args:
        shape: Shape of the output tensor.

    Returns:
        A tensor of Gumbel-distributed samples.
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel noise with the same shape, dtype, and device as ``x``.

    Args:
        x: Reference tensor whose shape, dtype, and device are matched.

    Returns:
        A tensor of Gumbel-distributed samples matching ``x``.
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Slice fixed-size segments from a batch of sequences at given start indices.

    Args:
        x: Input tensor of shape ``[B, C, T]``.
        ids_str: 1-D tensor of start indices, one per batch element.
        segment_size: Number of time steps in each segment.

    Returns:
        Tensor of shape ``[B, C, segment_size]`` containing the sliced segments.
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice fixed-size segments from a batch of sequences.

    Args:
        x: Input tensor of shape ``[B, C, T]``.
        x_lengths: Optional 1-D tensor of valid sequence lengths per batch element.
            If ``None``, the full time dimension ``T`` is used for all elements.
        segment_size: Number of time steps in each segment.

    Returns:
        A tuple ``(segments, ids_str)`` where ``segments`` has shape
        ``[B, C, segment_size]`` and ``ids_str`` contains the sampled start indices.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
    length: int,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Generate a sinusoidal positional encoding signal.

    Args:
        length: Number of time steps.
        channels: Number of channels (embedding dimension).
        min_timescale: Minimum timescale for the sinusoidal encoding.
        max_timescale: Maximum timescale for the sinusoidal encoding.

    Returns:
        Positional signal tensor of shape ``[1, channels, length]``.
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Add a sinusoidal positional encoding to a tensor.

    Args:
        x: Input tensor of shape ``[B, C, T]``.
        min_timescale: Minimum timescale for the positional encoding.
        max_timescale: Maximum timescale for the positional encoding.

    Returns:
        Tensor of the same shape as ``x`` with positional encoding added.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    """Concatenate a sinusoidal positional encoding to a tensor along a given axis.

    Args:
        x: Input tensor of shape ``[B, C, T]``.
        min_timescale: Minimum timescale for the positional encoding.
        max_timescale: Maximum timescale for the positional encoding.
        axis: Dimension along which to concatenate the signal.

    Returns:
        Tensor with the positional encoding concatenated along ``axis``.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    """Create a causal (lower-triangular) mask for self-attention.

    Args:
        length: Sequence length.

    Returns:
        Boolean mask tensor of shape ``[1, 1, length, length]`` where the upper
        triangle is ``0`` and the lower triangle (including diagonal) is ``1``.
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """Fused gated activation: tanh(a[:n]) * sigmoid(a[n:]) where a = input_a + input_b.

    This is the gating operation used in WaveNet-style architectures. Decorated
    with ``@torch.jit.script`` for performance.

    Args:
        input_a: First input tensor of shape ``[B, 2*n_channels, T]``.
        input_b: Second input tensor (e.g., conditioning) of shape ``[B, 2*n_channels, T]``.
        n_channels: 1-D tensor containing a single integer — the number of channels
            for the tanh branch.

    Returns:
        Gated activation tensor of shape ``[B, n_channels, T]``.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


# NOTE: duplicate definition kept for backward compatibility
def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:  # noqa: F811
    """Convert a nested list of pad shapes into a flat list suitable for F.pad.

    Reverses the order of the dimensions and flattens the nested structure,
    as required by PyTorch's F.pad which expects padding in reverse dimension order.

    Note:
        This is a duplicate of the earlier ``convert_pad_shape`` definition.
        It is kept here for backward compatibility with code that may rely on
        its position in the module.

    Args:
        pad_shape: A list of [pad_left, pad_right] pairs, one per dimension,
            ordered from outermost to innermost dimension.

    Returns:
        A flat list of padding values in the format expected by F.pad.
    """
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """Shift a sequence one step to the right by prepending a zero frame.

    Args:
        x: Input tensor of shape ``[B, C, T]``.

    Returns:
        Shifted tensor of the same shape, with the last time step removed and
        a zero frame prepended.
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(
    length: torch.Tensor,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Create a boolean sequence mask from lengths.

    Args:
        length: 1-D tensor of sequence lengths.
        max_length: Maximum length for the mask. If ``None``, uses
            ``length.max()``.

    Returns:
        Boolean tensor of shape ``[B, max_length]`` where entry ``[i, j]``
        is ``True`` iff ``j < length[i]``.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate an alignment path from durations and a validity mask.

    Args:
        duration: Integer duration tensor of shape ``[B, 1, T_x]``.
        mask: Binary mask tensor of shape ``[B, 1, T_y, T_x]``.

    Returns:
        Path tensor of shape ``[B, 1, T_y, T_x]`` indicating which
        (output frame, input token) pairs are aligned.
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    clip_value: Optional[float],
    norm_type: float = 2,
) -> float:
    """Clip gradients by value and return the total gradient norm.

    Args:
        parameters: A tensor or iterable of tensors whose gradients will be clipped.
        clip_value: Maximum absolute value for gradients. If ``None``, no clipping
            is applied but the norm is still computed.
        norm_type: Type of the norm used to compute the total gradient norm.

    Returns:
        The total gradient norm as a Python float.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
