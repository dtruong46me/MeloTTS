"""Building-block neural network modules for MeloTTS.

This module provides reusable PyTorch nn.Module components used throughout the
MeloTTS architecture, including normalisation layers, convolutional blocks,
WaveNet-style networks, residual blocks, and normalising-flow layers
(coupling layers, ConvFlow, TransformerCouplingLayer, etc.).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Conv1d, functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from . import commons
from .attentions import Encoder
from .commons import get_padding, init_weights
from .transforms import piecewise_rational_quadratic_transform

# Slope used for LeakyReLU activations throughout this module.
LRELU_SLOPE: float = 0.1


class LayerNorm(nn.Module):
    """Channel-last layer normalisation applied to 1-D sequence tensors.

    Transposes the channel dimension to the last position before calling
    ``F.layer_norm`` and then transposes back, so the module is compatible
    with inputs of shape ``[B, C, T]``.

    Attributes:
        channels: Number of feature channels.
        eps: Small constant added to the denominator for numerical stability.
        gamma: Learnable scale parameter of shape ``(channels,)``.
        beta: Learnable shift parameter of shape ``(channels,)``.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        """Initialise LayerNorm.

        Args:
            channels: Number of feature channels.
            eps: Epsilon for numerical stability in layer normalisation.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalisation.

        Args:
            x: Input tensor of shape ``[B, C, T]``.

        Returns:
            Normalised tensor of the same shape as ``x``.
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    """Stack of Conv1d → LayerNorm → ReLU → Dropout layers with a residual projection.

    The first layer maps ``in_channels`` to ``hidden_channels``; subsequent
    layers keep ``hidden_channels``; a final 1×1 projection maps back to
    ``out_channels`` and is initialised to zero so the residual starts as an
    identity transform.

    Attributes:
        in_channels: Number of input channels.
        hidden_channels: Number of channels in the intermediate layers.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        n_layers: Total number of convolutional layers (must be > 1).
        p_dropout: Dropout probability.
        conv_layers: List of Conv1d layers.
        norm_layers: List of LayerNorm layers.
        relu_drop: Sequential ReLU + Dropout applied after each norm.
        proj: Final 1×1 projection initialised to zero.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        """Initialise ConvReluNorm.

        Args:
            in_channels: Number of input feature channels.
            hidden_channels: Number of hidden feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Convolution kernel size (used with ``padding=kernel_size//2``).
            n_layers: Number of convolutional layers (must be > 1).
            p_dropout: Dropout probability applied after each ReLU.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Apply the stacked conv-norm-relu layers with a residual projection.

        Args:
            x: Input tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.

        Returns:
            Output tensor of shape ``[B, out_channels, T]``, masked by ``x_mask``.
        """
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """Dilated and Depth-Separable Convolution stack.

    Each layer applies a dilated depthwise Conv1d followed by a pointwise
    Conv1d (1×1), with LayerNorm and GELU activations.  Dilation is set to
    ``kernel_size ** layer_index`` so the receptive field grows exponentially.

    Attributes:
        channels: Number of feature channels (kept constant throughout).
        kernel_size: Base kernel size; dilation grows as ``kernel_size**i``.
        n_layers: Number of stacked layers.
        p_dropout: Dropout probability.
        drop: Dropout module.
        convs_sep: Depthwise (groups=channels) dilated convolutions.
        convs_1x1: Pointwise convolutions.
        norms_1: LayerNorm after depthwise conv.
        norms_2: LayerNorm after pointwise conv.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ) -> None:
        """Initialise DDSConv.

        Args:
            channels: Number of feature channels.
            kernel_size: Base convolution kernel size.
            n_layers: Number of DDSConv layers.
            p_dropout: Dropout probability.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the DDSConv stack.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor of shape ``[B, C, T]`` (or
               broadcastable). Added to ``x`` before the main loop.

        Returns:
            Output tensor of shape ``[B, C, T]``, masked by ``x_mask``.
        """
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(torch.nn.Module):
    """WaveNet-style stack of dilated gated convolutions with weight normalisation.

    Each layer uses a gated activation (tanh ⊗ sigmoid) conditioned on an
    optional global feature ``g``.  Skip connections are accumulated and
    returned as the output.

    Attributes:
        hidden_channels: Number of hidden feature channels.
        kernel_size: Convolution kernel size (stored as a tuple).
        dilation_rate: Base dilation; layer ``i`` uses ``dilation_rate**i``.
        n_layers: Number of WaveNet layers.
        gin_channels: Number of conditioning channels (0 = no conditioning).
        p_dropout: Dropout probability.
        in_layers: ModuleList of weight-normed dilated Conv1d layers.
        res_skip_layers: ModuleList of weight-normed 1×1 Conv1d layers.
        drop: Dropout module.
        cond_layer: Optional weight-normed Conv1d for global conditioning.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: int = 0,
    ) -> None:
        """Initialise WN (WaveNet-style) module.

        Args:
            hidden_channels: Number of hidden feature channels.
            kernel_size: Convolution kernel size (must be odd).
            dilation_rate: Base dilation factor; layer ``i`` has dilation
                ``dilation_rate**i``.
            n_layers: Number of stacked WaveNet layers.
            gin_channels: Dimension of the global conditioning vector.  Set to
                0 to disable conditioning.
            p_dropout: Dropout probability applied after gated activation.
        """
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the WaveNet forward pass.

        Args:
            x: Input tensor of shape ``[B, hidden_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor of shape
               ``[B, gin_channels, T']`` (or ``[B, gin_channels, 1]``).
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            Output tensor of shape ``[B, hidden_channels, T]``, masked.
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """Remove weight normalisation from all internal Conv1d layers.

        Should be called after training is complete, before inference, to
        fuse the normalisation into the weight tensors.
        """
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class ResBlock1(torch.nn.Module):
    """Residual block with multi-dilation Conv1d pairs (variant 1).

    Uses two parallel sets of three dilated Conv1d layers (``convs1`` and
    ``convs2``).  Each pair applies LeakyReLU, optional masking, and two
    convolutions with a residual connection.  All convolutions are
    weight-normalised.

    Attributes:
        convs1: ModuleList of weight-normed Conv1d with varying dilation.
        convs2: ModuleList of weight-normed Conv1d with dilation=1.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        """Initialise ResBlock1.

        Args:
            channels: Number of input and output channels.
            kernel_size: Convolution kernel size.
            dilation: Tuple of three dilation values for the first conv set.
        """
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            x_mask: Optional binary mask of shape ``[B, 1, T]``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalisation from all Conv1d layers in this block."""
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(torch.nn.Module):
    """Residual block with two dilated Conv1d layers (variant 2).

    Lighter than :class:`ResBlock1`; uses a single list of two convolutions
    with different dilation values.  All convolutions are weight-normalised.

    Attributes:
        convs: ModuleList of two weight-normed Conv1d layers.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ) -> None:
        """Initialise ResBlock2.

        Args:
            channels: Number of input and output channels.
            kernel_size: Convolution kernel size.
            dilation: Tuple of two dilation values.
        """
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            x_mask: Optional binary mask of shape ``[B, 1, T]``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalisation from all Conv1d layers in this block."""
        for layer in self.convs:
            remove_weight_norm(layer)


class Log(nn.Module):
    """Log-transform flow layer.

    In the forward direction computes ``y = log(clamp(x, min=1e-5))`` and
    returns the log-determinant.  In the reverse direction computes
    ``x = exp(y)``.

    This is a parameter-free bijection used inside normalising-flow chains.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply the log (forward) or exp (reverse) transform.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            reverse: If ``False`` (default) apply the forward (log) transform
                and return ``(y, logdet)``.  If ``True`` apply the inverse
                (exp) transform and return ``x``.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            In forward mode: a tuple ``(y, logdet)`` where ``y`` has the same
            shape as ``x`` and ``logdet`` is a scalar per sample.
            In reverse mode: the reconstructed ``x`` tensor.
        """
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    """Channel-flip flow layer.

    Reverses the channel dimension (``torch.flip(x, [1])``).  This is a
    parameter-free involution (its own inverse), so both the forward and
    reverse passes perform the same operation.  In the forward direction a
    zero log-determinant is also returned.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Flip channels and optionally return the log-determinant.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            *args: Ignored positional arguments.
            reverse: If ``False`` (default) return ``(x_flipped, logdet)``
                where ``logdet`` is all zeros.  If ``True`` return only the
                flipped tensor.
            **kwargs: Ignored keyword arguments.

        Returns:
            In forward mode: ``(x_flipped, logdet)`` — a zero log-determinant
            tensor of shape ``[B]``.
            In reverse mode: ``x_flipped``.
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """Learnable element-wise affine flow layer.

    Applies the transformation ``y = m + exp(logs) * x`` in the forward
    direction and ``x = (y - m) * exp(-logs)`` in the reverse direction.
    Both ``m`` and ``logs`` are learnable parameters.

    Attributes:
        channels: Number of feature channels.
        m: Learnable shift parameter of shape ``(channels, 1)``.
        logs: Learnable log-scale parameter of shape ``(channels, 1)``.
    """

    def __init__(self, channels: int) -> None:
        """Initialise ElementwiseAffine.

        Args:
            channels: Number of feature channels.
        """
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply element-wise affine transform.

        Args:
            x: Input tensor of shape ``[B, channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            reverse: If ``False`` (default) apply forward transform and return
                ``(y, logdet)``.  If ``True`` apply inverse and return ``x``.
            **kwargs: Ignored.

        Returns:
            In forward mode: ``(y, logdet)`` where ``logdet`` is a scalar
            per sample.
            In reverse mode: the reconstructed ``x`` tensor.
        """
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    """Affine coupling layer for normalising flows using a WaveNet encoder.

    Splits the input into two halves along the channel axis.  The first half
    is encoded by a WN network to produce mean (and optionally log-scale)
    parameters, which are applied to the second half.

    Attributes:
        channels: Total number of channels (must be even).
        hidden_channels: Hidden channels in the WN encoder.
        kernel_size: Kernel size for the WN convolutions.
        dilation_rate: Dilation rate for the WN convolutions.
        n_layers: Number of WN layers.
        half_channels: ``channels // 2``.
        mean_only: If ``True`` only predict the mean (log-scale is zero).
        pre: 1×1 Conv1d projecting the first half to ``hidden_channels``.
        enc: WN encoder network.
        post: 1×1 Conv1d projecting to the mean/log-scale output.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        """Initialise ResidualCouplingLayer.

        Args:
            channels: Total channel count (must be divisible by 2).
            hidden_channels: Number of channels in the WN encoder.
            kernel_size: Kernel size for WN dilated convolutions.
            dilation_rate: Dilation rate base for WN layers.
            n_layers: Number of layers in the WN encoder.
            p_dropout: Dropout probability for the WN encoder.
            gin_channels: Conditioning channel size (0 = none).
            mean_only: If ``True`` predict only the mean; log-scale is fixed
                to zero (volume-preserving flow).
        """
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply the residual coupling layer.

        Args:
            x: Input tensor of shape ``[B, channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor.
            reverse: If ``False`` (default) apply the forward coupling and
                return ``(x_out, logdet)``.  If ``True`` apply the inverse
                and return ``x_out``.

        Returns:
            In forward mode: ``(x_out, logdet)`` where ``logdet`` is scalar
            per sample.
            In reverse mode: reconstructed input tensor.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ConvFlow(nn.Module):
    """Coupling flow using a piecewise rational quadratic spline transform.

    The first half of the channels is encoded by a :class:`DDSConv` network
    to produce spline knot parameters, which are applied to the second half
    via a piecewise rational quadratic transform.

    Attributes:
        in_channels: Total number of input channels.
        filter_channels: Number of channels in the DDSConv encoder.
        kernel_size: Kernel size for the DDSConv encoder.
        n_layers: Number of DDSConv layers.
        num_bins: Number of spline bins.
        tail_bound: Boundary of the spline tails.
        half_channels: ``in_channels // 2``.
        pre: 1×1 Conv1d projecting the first half to ``filter_channels``.
        convs: DDSConv encoder.
        proj: 1×1 Conv1d projecting to knot parameters.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        """Initialise ConvFlow.

        Args:
            in_channels: Total number of input channels (must be even).
            filter_channels: Number of channels in the internal DDSConv.
            kernel_size: Kernel size for DDSConv.
            n_layers: Number of DDSConv layers.
            num_bins: Number of rational quadratic spline bins.
            tail_bound: Boundary value for the linear tails of the spline.
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply the convolutional spline flow.

        Args:
            x: Input tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor passed to DDSConv.
            reverse: If ``False`` (default) apply forward transform and return
                ``(x_out, logdet)``.  If ``True`` apply inverse and return
                ``x_out``.

        Returns:
            In forward mode: ``(x_out, logdet)``.
            In reverse mode: ``x_out``.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class TransformerCouplingLayer(nn.Module):
    """Affine coupling layer using a Transformer encoder instead of WaveNet.

    Structurally identical to :class:`ResidualCouplingLayer` but replaces the
    WN encoder with an :class:`~melo.attentions.Encoder` (Transformer FFT
    block).  Supports parameter sharing via ``wn_sharing_parameter``.

    Attributes:
        channels: Total number of channels (must be even).
        hidden_channels: Hidden channels in the Transformer encoder.
        kernel_size: Kernel size for the Transformer feed-forward network.
        n_layers: Number of Transformer layers (must be 3).
        half_channels: ``channels // 2``.
        mean_only: If ``True`` only predict the mean (log-scale is zero).
        pre: 1×1 Conv1d projecting the first half to ``hidden_channels``.
        enc: Transformer Encoder (or shared parameter reference).
        post: 1×1 Conv1d projecting to mean/log-scale output.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0,
        filter_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter: Optional[nn.Module] = None,
        gin_channels: int = 0,
    ) -> None:
        """Initialise TransformerCouplingLayer.

        Args:
            channels: Total channel count (must be divisible by 2).
            hidden_channels: Number of hidden channels in the Transformer encoder.
            kernel_size: Kernel size for the Transformer FFT feed-forward layers.
            n_layers: Number of Transformer layers (must equal 3).
            n_heads: Number of attention heads.
            p_dropout: Dropout probability.
            filter_channels: Feed-forward filter channel count for the Transformer.
            mean_only: If ``True`` predict only the mean (volume-preserving).
            wn_sharing_parameter: If not ``None``, this encoder module is reused
                instead of creating a new one (parameter sharing across layers).
            gin_channels: Global conditioning channel size (0 = none).
        """
        assert n_layers == 3, n_layers
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = (
            Encoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply the Transformer coupling layer.

        Args:
            x: Input tensor of shape ``[B, channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor.
            reverse: If ``False`` (default) apply forward coupling and return
                ``(x_out, logdet)``.  If ``True`` apply inverse and return
                ``x_out``.

        Returns:
            In forward mode: ``(x_out, logdet)`` where ``logdet`` is scalar
            per sample.
            In reverse mode: reconstructed input tensor.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

        # NOTE: unreachable code below - kept for reference
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x
