"""Attention mechanisms for MeloTTS.

This module implements the core attention building blocks used throughout
the MeloTTS architecture:

- **LayerNorm**: Channel-first Layer Normalization wrapper.
- **Encoder**: Transformer encoder stack with optional speaker-conditioning.
- **Decoder**: Transformer decoder stack with self-attention and
  encoder-decoder cross-attention.
- **MultiHeadAttention**: Scaled dot-product multi-head attention with
  optional relative-position embeddings and proximal bias.
- **FFN**: Position-wise feed-forward network with causal or same padding.
- **fused_add_tanh_sigmoid_multiply**: TorchScript-fused gating activation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from . import commons

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """Channel-first Layer Normalization.

    Applies ``torch.nn.functional.layer_norm`` along the channel dimension
    of tensors shaped ``[B, C, T]`` by temporarily transposing to
    ``[B, T, C]``.

    Attributes:
        channels: Number of channels (features).
        eps: Small value added to the denominator for numerical stability.
        gamma: Learnable scale parameter of shape ``(channels,)``.
        beta: Learnable shift parameter of shape ``(channels,)``.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        """Initialise LayerNorm.

        Args:
            channels: Number of feature channels.
            eps: Epsilon for numerical stability in the normalisation
                denominator.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-first layer normalisation.

        Args:
            x: Input tensor of shape ``[B, C, T]``.

        Returns:
            Normalised tensor of the same shape ``[B, C, T]``.
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """Fused gating activation: tanh * sigmoid on summed inputs.

    Splits the summed activation along the channel axis into two halves,
    applies ``tanh`` to the first half and ``sigmoid`` to the second, then
    returns their element-wise product.  Decorated with
    ``@torch.jit.script`` for fused kernel execution.

    Args:
        input_a: First input tensor of shape ``[B, 2*C, T]``.
        input_b: Second input tensor of shape ``[B, 2*C, T]``.
        n_channels: 1-D integer tensor whose first element is ``C``
            (the number of channels per gate half).

    Returns:
        Gated output tensor of shape ``[B, C, T]``.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Encoder(nn.Module):
    """Transformer encoder with optional speaker conditioning.

    Stacks ``n_layers`` of Multi-Head Self-Attention + Feed-Forward
    sub-layers, each followed by residual connection and Layer
    Normalization.  An optional speaker embedding can be injected at a
    configurable layer index via ``gin_channels`` / ``cond_layer_idx``
    keyword arguments.

    Attributes:
        hidden_channels: Dimensionality of hidden representations.
        filter_channels: Inner dimensionality of each FFN sub-layer.
        n_heads: Number of attention heads.
        n_layers: Total number of encoder layers.
        kernel_size: Convolution kernel size used in FFN.
        p_dropout: Dropout probability.
        window_size: Relative-attention window size (``None`` disables it).
        cond_layer_idx: Layer index at which speaker conditioning is added.
        drop: Dropout module.
        attn_layers: List of ``MultiHeadAttention`` modules.
        norm_layers_1: Layer norms after the attention sub-layers.
        ffn_layers: List of ``FFN`` modules.
        norm_layers_2: Layer norms after the FFN sub-layers.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        isflow: bool = True,
        **kwargs,
    ) -> None:
        """Initialise the Encoder.

        Args:
            hidden_channels: Dimensionality of hidden representations.
            filter_channels: Inner dimensionality of the FFN sub-layers.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers to stack.
            kernel_size: Convolution kernel size used inside FFN.
            p_dropout: Dropout probability applied after each sub-layer.
            window_size: Window size for relative-position attention
                embeddings.
            isflow: Unused flag kept for API compatibility.
            **kwargs: Optional keyword arguments:
                - ``gin_channels`` (int): Dimensionality of the speaker
                  embedding.  When non-zero a linear projection and
                  conditioning are enabled.
                - ``cond_layer_idx`` (int): Layer index at which the
                  speaker embedding is added (default ``2``).
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                assert (
                    self.cond_layer_idx < self.n_layers
                ), "cond_layer_idx should be less than n_layers"
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    filter_channels=filter_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the encoder forward pass.

        Args:
            x: Input tensor of shape ``[B, C, T]``.
            x_mask: Binary mask tensor of shape ``[B, 1, T]``.
            g: Optional speaker embedding of shape ``[B, gin_channels, 1]``.
                When provided, it is injected at ``cond_layer_idx``.

        Returns:
            Encoded output tensor of shape ``[B, C, T]``.
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Decoder(nn.Module):
    """Transformer decoder with self-attention and cross-attention.

    Stacks ``n_layers`` of:
    1. Masked Multi-Head Self-Attention (causal, with optional proximal bias).
    2. Multi-Head Encoder-Decoder Cross-Attention.
    3. Causal Feed-Forward Network.

    Each sub-layer is followed by a residual connection and Layer Norm.

    Attributes:
        hidden_channels: Dimensionality of hidden representations.
        filter_channels: Inner dimensionality of each FFN sub-layer.
        n_heads: Number of attention heads.
        n_layers: Total number of decoder layers.
        kernel_size: Convolution kernel size used in FFN.
        p_dropout: Dropout probability.
        proximal_bias: Whether to add proximal (locality) bias to
            self-attention scores.
        proximal_init: Whether to initialise key projection weights from
            query weights in self-attention.
        drop: Dropout module.
        self_attn_layers: Self-attention ``MultiHeadAttention`` modules.
        norm_layers_0: Layer norms after self-attention.
        encdec_attn_layers: Cross-attention ``MultiHeadAttention`` modules.
        norm_layers_1: Layer norms after cross-attention.
        ffn_layers: ``FFN`` modules (causal padding).
        norm_layers_2: Layer norms after FFN.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        **kwargs,
    ) -> None:
        """Initialise the Decoder.

        Args:
            hidden_channels: Dimensionality of hidden representations.
            filter_channels: Inner dimensionality of the FFN sub-layers.
            n_heads: Number of attention heads.
            n_layers: Number of decoder layers to stack.
            kernel_size: Convolution kernel size used inside FFN.
            p_dropout: Dropout probability applied after each sub-layer.
            proximal_bias: If ``True``, add a proximity bias to
                self-attention to encourage attending to nearby positions.
            proximal_init: If ``True``, initialise the self-attention key
                projection weights from the query projection weights.
            **kwargs: Absorbed for forward-compatibility.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    filter_channels=filter_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder forward pass.

        Args:
            x: Decoder input tensor of shape ``[B, C, T_dec]``.
            x_mask: Binary mask for decoder input, shape ``[B, 1, T_dec]``.
            h: Encoder output tensor of shape ``[B, C, T_enc]``.
            h_mask: Binary mask for encoder output, shape ``[B, 1, T_enc]``.

        Returns:
            Decoded output tensor of shape ``[B, C, T_dec]``.
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    """Scaled dot-product Multi-Head Attention with optional extensions.

    Supports:
    - Relative-position embeddings (Music Transformer style) when
      ``window_size`` is set.
    - Proximal (locality) bias to bias attention toward nearby tokens.
    - Block/local attention via ``block_length``.
    - Causal initialisation where key weights copy query weights
      (``proximal_init``).

    Attributes:
        channels: Input channel dimensionality.
        out_channels: Output channel dimensionality.
        n_heads: Number of attention heads.
        p_dropout: Dropout probability on attention weights.
        window_size: Half-width of relative-position window
            (``None`` disables relative embeddings).
        heads_share: Whether all heads share a single set of relative
            position embeddings.
        block_length: Maximum attention span for local/block attention
            (``None`` disables).
        proximal_bias: Whether to add log-inverse-distance proximity bias.
        proximal_init: Whether to initialise ``conv_k`` weights from
            ``conv_q``.
        attn: Stores the most recent attention probability map (set during
            ``forward``).
        k_channels: Per-head key/query/value dimensionality.
        conv_q: Query projection ``Conv1d``.
        conv_k: Key projection ``Conv1d``.
        conv_v: Value projection ``Conv1d``.
        conv_o: Output projection ``Conv1d``.
        drop: Dropout applied to attention probabilities.
        emb_rel_k: Relative-position key embeddings (when enabled).
        emb_rel_v: Relative-position value embeddings (when enabled).
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        """Initialise MultiHeadAttention.

        Args:
            channels: Input channel dimensionality.  Must be divisible by
                ``n_heads``.
            out_channels: Output channel dimensionality.
            n_heads: Number of parallel attention heads.
            p_dropout: Dropout probability on attention weights.
            window_size: When set, enables relative-position embeddings
                with a window of ``[-window_size, window_size]``.
            heads_share: If ``True``, all heads share one set of relative
                embeddings; otherwise each head has its own.
            block_length: If set, restricts attention to a local window of
                ±``block_length`` positions.
            proximal_bias: If ``True``, adds a log-inverse-distance bias
                to self-attention scores.
            proximal_init: If ``True``, copies ``conv_q`` weights to
                ``conv_k`` at initialisation (useful for proximal attention
                in decoders).
        """
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention output.

        Projects ``x`` to queries and ``c`` to keys/values, then calls
        :meth:`attention` and projects the result to the output space.

        Args:
            x: Query source tensor of shape ``[B, C, T_q]``.
            c: Key/value source tensor of shape ``[B, C, T_k]``.
            attn_mask: Optional boolean mask of shape
                ``[B, 1, T_q, T_k]`` (or broadcastable).  Positions with
                value ``0`` are masked out (set to ``-1e4`` before softmax).

        Returns:
            Output tensor of shape ``[B, out_channels, T_q]``.
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with optional extensions.

        Reshapes query/key/value from ``[B, D, T]`` to
        ``[B, n_heads, T, d_k]``, computes attention scores, optionally
        adds relative-position logits and/or proximal bias, masks,
        softmaxes, and returns the attended output together with the
        attention probability map.

        Args:
            query: Query tensor of shape ``[B, D, T_q]``.
            key: Key tensor of shape ``[B, D, T_k]``.
            value: Value tensor of shape ``[B, D, T_k]``.
            mask: Optional mask of shape broadcastable to
                ``[B, n_heads, T_q, T_k]``.  Zero entries are masked.

        Returns:
            A tuple of:
            - output: Attended tensor of shape ``[B, D, T_q]``.
            - p_attn: Attention probabilities of shape
              ``[B, n_heads, T_q, T_k]``.
        """
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (
                t_s == t_t
            ), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (
                    t_s == t_t
                ), "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Batched matrix multiply of attention weights with relative values.

        Args:
            x: Attention weights of shape ``[b, h, l, m]``.
            y: Relative value embeddings of shape ``[h or 1, m, d]``.

        Returns:
            Result tensor of shape ``[b, h, l, d]``.
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Batched matrix multiply of queries with relative key embeddings.

        Args:
            x: Scaled query tensor of shape ``[b, h, l, d]``.
            y: Relative key embeddings of shape ``[h or 1, m, d]``.

        Returns:
            Relative logit tensor of shape ``[b, h, l, m]``.
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(
        self,
        relative_embeddings: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Slice or pad relative embeddings to match the current sequence length.

        Pads first (to avoid conditional ops during tracing), then slices
        to the window of size ``2*length - 1`` centred on the current
        position.

        Args:
            relative_embeddings: Stored embedding table of shape
                ``[h or 1, 2*window_size+1, d_k]``.
            length: Current sequence length ``T``.

        Returns:
            Embeddings of shape ``[h or 1, 2*T-1, d_k]``.
        """
        2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convert relative-position logits to absolute-position scores.

        Uses a skewing trick (padding + reshape) to go from relative to
        absolute indexing without any conditional operations.

        Args:
            x: Relative-position logit tensor of shape
                ``[b, h, l, 2*l-1]``.

        Returns:
            Absolute-position score tensor of shape ``[b, h, l, l]``.
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1:
        ]
        return x_final

    def _absolute_position_to_relative_position(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convert absolute-position attention weights to relative indexing.

        Uses a skewing trick (padding + reshape) to go from absolute to
        relative indexing, producing the ``2*l-1`` relative positions.

        Args:
            x: Absolute-position attention weight tensor of shape
                ``[b, h, l, l]``.

        Returns:
            Relative-position weight tensor of shape ``[b, h, l, 2*l-1]``.
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """Bias for self-attention to encourage attention to close positions.

        Computes ``-log(1 + |i - j|)`` for all position pairs ``(i, j)``,
        producing a bias that penalises attending to distant positions.

        Args:
            length: Sequence length (integer scalar).

        Returns:
            Bias tensor of shape ``[1, 1, length, length]``.
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """Position-wise Feed-Forward Network.

    Applies two 1-D convolutions with an intermediate activation (ReLU
    or GELU approximation), dropout, and optional causal or same padding.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        filter_channels: Number of intermediate channels.
        kernel_size: Convolution kernel size.
        p_dropout: Dropout probability.
        activation: Activation type (``None`` → ReLU, ``"gelu"`` → GELU).
        causal: Whether to use causal (left-only) padding.
        padding: Bound method for padding (either :meth:`_causal_padding`
            or :meth:`_same_padding`).
        conv_1: First ``Conv1d`` layer.
        conv_2: Second ``Conv1d`` layer.
        drop: Dropout module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ) -> None:
        """Initialise FFN.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            filter_channels: Number of intermediate (hidden) channels.
            kernel_size: Size of the 1-D convolution kernel.
            p_dropout: Dropout probability applied between the two
                convolutions.
            activation: Activation function selector.  ``None`` uses ReLU;
                ``"gelu"`` uses a sigmoid-based GELU approximation.
            causal: If ``True``, use causal (left-only) padding so that
                position ``t`` only sees positions ``<= t``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation.

        Args:
            x: Input tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask tensor of shape ``[B, 1, T]``.

        Returns:
            Output tensor of shape ``[B, out_channels, T]``, masked by
            ``x_mask``.
        """
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal (left-only) padding to the input tensor.

        Pads ``kernel_size - 1`` elements on the left and zero on the
        right so that each output position only depends on past inputs.

        Args:
            x: Input tensor of shape ``[B, C, T]``.

        Returns:
            Padded tensor of shape ``[B, C, T + kernel_size - 1]``,
            or ``x`` unchanged when ``kernel_size == 1``.
        """
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply symmetric (same) padding to keep the sequence length.

        Pads ``(kernel_size - 1) // 2`` on the left and
        ``kernel_size // 2`` on the right.

        Args:
            x: Input tensor of shape ``[B, C, T]``.

        Returns:
            Padded tensor of shape ``[B, C, T + kernel_size - 1]``,
            or ``x`` unchanged when ``kernel_size == 1``.
        """
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
