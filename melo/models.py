"""High-level MeloTTS model definitions.

This module contains all major neural network architectures used in MeloTTS:

* ``DurationDiscriminator`` — VITS2-style duration discriminator.
* ``TransformerCouplingBlock`` — normalising-flow block using Transformer layers.
* ``StochasticDurationPredictor`` — flow-based stochastic duration predictor.
* ``DurationPredictor`` — deterministic duration predictor.
* ``TextEncoder`` — phoneme/tone/language/BERT text encoder.
* ``ResidualCouplingBlock`` — WaveNet-based residual coupling flow.
* ``PosteriorEncoder`` — posterior encoder for the latent variable.
* ``Generator`` — HiFi-GAN-style waveform decoder.
* ``DiscriminatorP`` / ``DiscriminatorS`` / ``MultiPeriodDiscriminator``
  — discriminator stack for adversarial training.
* ``ReferenceEncoder`` — reference-audio encoder for speaker conditioning.
* ``SynthesizerTrn`` — end-to-end synthesiser (training and inference).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d, functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from melo import attentions, commons, modules
from melo.commons import get_padding, init_weights
import melo.monotonic_align as monotonic_align

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Dimension of the multilingual BERT embedding (e.g. bert-base-multilingual).
BERT_DIM: int = 1024

# Dimension of the Japanese BERT embedding (e.g. cl-tohoku/bert-base-japanese).
JA_BERT_DIM: int = 768


# ---------------------------------------------------------------------------
# DurationDiscriminator
# ---------------------------------------------------------------------------


class DurationDiscriminator(nn.Module):  # vits2
    """VITS2 duration discriminator.

    Scores the plausibility of real vs. predicted duration sequences.
    Accepts two duration tensors (real and predicted) and returns one
    probability per duration via a shared convolutional feature extractor
    followed by ``forward_probability``.

    Attributes:
        in_channels: Input feature channel count.
        filter_channels: Intermediate filter channel count.
        kernel_size: Convolution kernel size.
        p_dropout: Dropout probability.
        gin_channels: Global conditioning channel size (0 = none).
        drop: Dropout module.
        conv_1: First Conv1d.
        norm_1: LayerNorm after first conv.
        conv_2: Second Conv1d.
        norm_2: LayerNorm after second conv.
        dur_proj: 1×1 Conv1d projecting the duration scalar to filter_channels.
        pre_out_conv_1: First conv in the probability head.
        pre_out_norm_1: LayerNorm after pre_out_conv_1.
        pre_out_conv_2: Second conv in the probability head.
        pre_out_norm_2: LayerNorm after pre_out_conv_2.
        cond: Optional conditioning Conv1d (present when gin_channels > 0).
        output_layer: Linear → Sigmoid to produce [0, 1] probability.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        """Initialise DurationDiscriminator.

        Args:
            in_channels: Number of input feature channels.
            filter_channels: Number of intermediate filter channels.
            kernel_size: Convolution kernel size.
            p_dropout: Dropout probability.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-frame output probabilities given features and a duration.

        Args:
            x: Feature tensor of shape ``[B, filter_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            dur: Duration tensor of shape ``[B, 1, T]``.
            g: Unused; kept for API consistency.

        Returns:
            Probability tensor of shape ``[B, T, 1]``.
        """
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur_r: torch.Tensor,
        dur_hat: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Discriminate real vs. predicted durations.

        Args:
            x: Input feature tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            dur_r: Real duration tensor of shape ``[B, 1, T]``.
            dur_hat: Predicted duration tensor of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor of shape
               ``[B, gin_channels, 1]``.

        Returns:
            List of two probability tensors ``[prob_real, prob_fake]``, each
            of shape ``[B, T, 1]``.
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs


# ---------------------------------------------------------------------------
# TransformerCouplingBlock
# ---------------------------------------------------------------------------


class TransformerCouplingBlock(nn.Module):
    """Normalising-flow block composed of Transformer coupling layers.

    Stacks ``n_flows`` :class:`~melo.modules.TransformerCouplingLayer` layers
    interleaved with :class:`~melo.modules.Flip` layers.  Supports optional
    parameter sharing across coupling layers.

    Attributes:
        channels: Feature channel count.
        hidden_channels: Hidden channel count.
        kernel_size: Convolution kernel size in the feed-forward network.
        n_layers: Number of Transformer layers per coupling layer.
        n_flows: Number of coupling-layer + flip pairs.
        gin_channels: Global conditioning channel size.
        flows: ModuleList of alternating coupling layers and flips.
        wn: Shared Transformer FFT encoder (or ``None``).
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        """Initialise TransformerCouplingBlock.

        Args:
            channels: Number of feature channels.
            hidden_channels: Number of hidden channels in each coupling layer.
            filter_channels: Feed-forward filter channels in the Transformer.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers (must be 3).
            kernel_size: Convolution kernel size for the feed-forward network.
            p_dropout: Dropout probability.
            n_flows: Number of flow steps (coupling layer + flip pairs).
            gin_channels: Global conditioning channel size (0 = none).
            share_parameter: If ``True``, all coupling layers share the same
                Transformer encoder weights.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Apply (or invert) the Transformer coupling flow.

        Args:
            x: Input tensor of shape ``[B, channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor.
            reverse: If ``False`` (default) apply forward flows in order.
                If ``True`` apply flows in reversed order.

        Returns:
            Transformed tensor of the same shape as ``x``.
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ---------------------------------------------------------------------------
# StochasticDurationPredictor
# ---------------------------------------------------------------------------


class StochasticDurationPredictor(nn.Module):
    """Flow-based stochastic duration predictor.

    Models the conditional distribution of phoneme durations using a
    normalising flow (ConvFlow + Flip chain) with a variational posterior
    trained via the flow-based objective.

    Attributes:
        in_channels: Input feature channel count.
        filter_channels: Internal filter channel count (overridden to
            ``in_channels`` for the current version).
        kernel_size: Convolution kernel size for DDSConv.
        p_dropout: Dropout probability.
        n_flows: Number of ConvFlow + Flip pairs in the main flow.
        gin_channels: Global conditioning channel size (0 = none).
        log_flow: Log-transform flow layer.
        flows: Main normalising-flow chain.
        post_pre / post_convs / post_proj: Posterior network layers.
        post_flows: Posterior normalising-flow chain.
        pre / convs / proj: Prior network layers.
        cond: Optional conditioning Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """Initialise StochasticDurationPredictor.

        Args:
            in_channels: Number of input feature channels.
            filter_channels: Filter channel count (overridden to ``in_channels``
                in the current implementation — kept for API compatibility).
            kernel_size: Kernel size for DDSConv layers.
            p_dropout: Dropout probability.
            n_flows: Number of ConvFlow + Flip pairs.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass for training (NLL) or inference (log-duration).

        Args:
            x: Input feature tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            w: Target duration tensor of shape ``[B, 1, T]``.  Required in
               forward (non-reverse) mode.
            g: Optional global conditioning tensor.
            reverse: If ``False`` (default) return the negative log-likelihood
               ``nll + logq`` for training.  If ``True`` return ``logw``
               (predicted log-duration) for inference.
            noise_scale: Scale applied to the Gaussian noise in reverse mode.

        Returns:
            In forward mode: NLL scalar per sample of shape ``[B]``.
            In reverse mode: Log-duration tensor ``logw`` of shape
            ``[B, 1, T]``.
        """
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


# ---------------------------------------------------------------------------
# DurationPredictor
# ---------------------------------------------------------------------------


class DurationPredictor(nn.Module):
    """Deterministic phoneme duration predictor.

    Two-layer Conv1d network that maps phoneme features to log-durations.

    Attributes:
        in_channels: Input feature channel count.
        filter_channels: Intermediate filter channel count.
        kernel_size: Convolution kernel size.
        p_dropout: Dropout probability.
        gin_channels: Global conditioning channel size (0 = none).
        drop: Dropout module.
        conv_1: First Conv1d.
        norm_1: LayerNorm after first conv.
        conv_2: Second Conv1d.
        norm_2: LayerNorm after second conv.
        proj: 1×1 Conv1d projecting to a single scalar per time step.
        cond: Optional conditioning Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        """Initialise DurationPredictor.

        Args:
            in_channels: Number of input feature channels.
            filter_channels: Number of intermediate filter channels.
            kernel_size: Convolution kernel size.
            p_dropout: Dropout probability.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict log-durations.

        Args:
            x: Input feature tensor of shape ``[B, in_channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor.

        Returns:
            Log-duration tensor of shape ``[B, 1, T]``, masked.
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Phoneme / tone / language / BERT text encoder.

    Fuses phoneme embeddings, tone embeddings, language embeddings, and
    multilingual BERT features (both general and Japanese) before passing
    them through a Transformer encoder to produce mean and log-variance
    parameters for the prior distribution.

    Attributes:
        n_vocab: Phoneme vocabulary size.
        out_channels: Output channel count (half of the projection).
        hidden_channels: Hidden channel count.
        filter_channels: Feed-forward filter channels in the Transformer.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        kernel_size: Convolution kernel size in the feed-forward network.
        p_dropout: Dropout probability.
        gin_channels: Global conditioning channel size.
        emb: Phoneme embedding table.
        tone_emb: Tone embedding table.
        language_emb: Language ID embedding table.
        bert_proj: 1×1 Conv1d projecting BERT_DIM → hidden_channels.
        ja_bert_proj: 1×1 Conv1d projecting JA_BERT_DIM → hidden_channels.
        encoder: Transformer encoder.
        proj: 1×1 Conv1d projecting hidden_channels → out_channels * 2.
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
        num_languages: Optional[int] = None,
        num_tones: Optional[int] = None,
    ) -> None:
        """Initialise TextEncoder.

        Args:
            n_vocab: Phoneme vocabulary size.
            out_channels: Output channel count for mean/log-variance.
            hidden_channels: Hidden channel count in the Transformer.
            filter_channels: Feed-forward filter channels in the Transformer.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers.
            kernel_size: Convolution kernel size.
            p_dropout: Dropout probability.
            gin_channels: Global conditioning channel size (0 = none).
            num_languages: Number of language IDs.  If ``None``, imported
                from ``text``.
            num_tones: Number of tone IDs.  If ``None``, imported from
                ``text``.
        """
        super().__init__()
        if num_languages is None:
            from text import num_languages
        if num_tones is None:
            from text import num_tones
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        self.ja_bert_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        ja_bert: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text inputs into mean and log-variance for the prior.

        Args:
            x: Phoneme ID tensor of shape ``[B, T]``.
            x_lengths: Sequence length tensor of shape ``[B]``.
            tone: Tone ID tensor of shape ``[B, T]``.
            language: Language ID tensor of shape ``[B, T]``.
            bert: BERT feature tensor of shape ``[B, BERT_DIM, T]``.
            ja_bert: Japanese BERT feature tensor of shape
               ``[B, JA_BERT_DIM, T]``.
            g: Optional global conditioning tensor.

        Returns:
            Tuple of ``(x, m, logs, x_mask)`` where:
            - ``x`` is the encoded feature tensor ``[B, hidden_channels, T]``.
            - ``m`` is the prior mean ``[B, out_channels, T]``.
            - ``logs`` is the prior log-variance ``[B, out_channels, T]``.
            - ``x_mask`` is the binary mask ``[B, 1, T]``.
        """
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


# ---------------------------------------------------------------------------
# ResidualCouplingBlock
# ---------------------------------------------------------------------------


class ResidualCouplingBlock(nn.Module):
    """WaveNet-based residual coupling flow block.

    Stacks ``n_flows`` :class:`~melo.modules.ResidualCouplingLayer` layers
    interleaved with :class:`~melo.modules.Flip` layers.

    Attributes:
        channels: Feature channel count.
        hidden_channels: Hidden channel count in each coupling layer.
        kernel_size: Kernel size for WN convolutions.
        dilation_rate: Base dilation rate for WN convolutions.
        n_layers: Number of WN layers per coupling layer.
        n_flows: Number of flow steps.
        gin_channels: Global conditioning channel size.
        flows: ModuleList of alternating coupling layers and flips.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """Initialise ResidualCouplingBlock.

        Args:
            channels: Number of feature channels.
            hidden_channels: Number of hidden channels in each coupling layer.
            kernel_size: Kernel size for the WN dilated convolutions.
            dilation_rate: Base dilation rate for the WN convolutions.
            n_layers: Number of WN layers per coupling layer.
            n_flows: Number of coupling layer + flip pairs.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Apply (or invert) the residual coupling flow.

        Args:
            x: Input tensor of shape ``[B, channels, T]``.
            x_mask: Binary mask of shape ``[B, 1, T]``.
            g: Optional global conditioning tensor.
            reverse: If ``False`` (default) apply forward flows in order.
                If ``True`` apply flows in reversed order.

        Returns:
            Transformed tensor of the same shape as ``x``.
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# ---------------------------------------------------------------------------
# PosteriorEncoder
# ---------------------------------------------------------------------------


class PosteriorEncoder(nn.Module):
    """Posterior encoder mapping a mel-spectrogram to the latent variable z.

    Uses a WN (WaveNet-style) encoder to produce the mean and log-variance of
    the posterior distribution, then samples ``z`` using the reparameterisation
    trick (with optional temperature ``tau``).

    Attributes:
        in_channels: Input channel count (e.g. number of mel bins).
        out_channels: Latent dimension.
        hidden_channels: Hidden channel count in the WN encoder.
        kernel_size: Kernel size for WN convolutions.
        dilation_rate: Base dilation rate for WN convolutions.
        n_layers: Number of WN layers.
        gin_channels: Global conditioning channel size.
        pre: 1×1 Conv1d projecting in_channels → hidden_channels.
        enc: WN encoder.
        proj: 1×1 Conv1d projecting hidden_channels → out_channels * 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        """Initialise PosteriorEncoder.

        Args:
            in_channels: Number of input channels (e.g. mel-spectrogram bins).
            out_channels: Dimensionality of the latent variable ``z``.
            hidden_channels: Number of hidden channels in the WN encoder.
            kernel_size: Kernel size for WN dilated convolutions.
            dilation_rate: Base dilation rate for WN convolutions.
            n_layers: Number of WN layers.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode spectrogram into the latent posterior.

        Args:
            x: Input spectrogram tensor of shape ``[B, in_channels, T]``.
            x_lengths: Sequence length tensor of shape ``[B]``.
            g: Optional global conditioning tensor.
            tau: Temperature for the reparameterisation sampling (default 1.0).

        Returns:
            Tuple ``(z, m, logs, x_mask)`` where:
            - ``z`` is the sampled latent of shape ``[B, out_channels, T]``.
            - ``m`` is the posterior mean ``[B, out_channels, T]``.
            - ``logs`` is the posterior log-variance ``[B, out_channels, T]``.
            - ``x_mask`` is the binary mask ``[B, 1, T]``.
        """
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(torch.nn.Module):
    """HiFi-GAN-style waveform generator (decoder).

    Upsamples the latent tensor to a waveform using a series of transposed
    convolutions followed by multi-receptive-field fusion residual blocks.

    Attributes:
        num_kernels: Number of residual block kernel sizes.
        num_upsamples: Number of upsampling stages.
        conv_pre: Initial 7×1 Conv1d.
        ups: ModuleList of weight-normed ConvTranspose1d layers.
        resblocks: ModuleList of ResBlock1 or ResBlock2 instances.
        conv_post: Final 7×1 Conv1d with bias=False.
        cond: Optional 1×1 Conv1d for global conditioning.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
    ) -> None:
        """Initialise Generator.

        Args:
            initial_channel: Number of input channels (latent dimension).
            resblock: Residual block variant: ``"1"`` for ResBlock1 or
                ``"2"`` for ResBlock2.
            resblock_kernel_sizes: List of kernel sizes for residual blocks.
            resblock_dilation_sizes: List of dilation tuples for residual
                blocks.
            upsample_rates: Stride for each ConvTranspose1d upsampling stage.
            upsample_initial_channel: Channel count at the start of upsampling.
            upsample_kernel_sizes: Kernel sizes for ConvTranspose1d stages.
            gin_channels: Global conditioning channel size (0 = none).
        """
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a waveform from the latent representation.

        Args:
            x: Latent tensor of shape ``[B, initial_channel, T]``.
            g: Optional global conditioning tensor of shape
               ``[B, gin_channels, 1]``.

        Returns:
            Waveform tensor of shape ``[B, 1, T']`` in the range ``[-1, 1]``.
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalisation from all upsampling and residual layers."""
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


# ---------------------------------------------------------------------------
# DiscriminatorP
# ---------------------------------------------------------------------------


class DiscriminatorP(torch.nn.Module):
    """Period discriminator operating on sub-sampled waveform periods.

    Reshapes the 1-D waveform to 2-D (period × frames) and applies a stack
    of 2-D convolutions to discriminate real from generated audio at a given
    period.

    Attributes:
        period: Period used for reshaping the waveform.
        use_spectral_norm: If ``True`` use spectral normalisation instead of
            weight normalisation.
        convs: Stack of Conv2d layers.
        conv_post: Final Conv2d layer.
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        """Initialise DiscriminatorP.

        Args:
            period: Waveform period for 2-D reshaping.
            kernel_size: Kernel size for the 2-D convolutions.
            stride: Stride for the 2-D convolutions.
            use_spectral_norm: If ``True`` use spectral norm; otherwise weight
                norm.
        """
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Discriminate the input waveform at the configured period.

        Args:
            x: Waveform tensor of shape ``[B, 1, T]``.

        Returns:
            Tuple ``(score, fmap)`` where ``score`` is the flattened logit
            tensor of shape ``[B, T']`` and ``fmap`` is a list of intermediate
            feature maps.
        """
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# ---------------------------------------------------------------------------
# DiscriminatorS
# ---------------------------------------------------------------------------


class DiscriminatorS(torch.nn.Module):
    """Scale discriminator operating on the raw waveform.

    Applies a stack of 1-D convolutions at the full waveform resolution to
    discriminate real from generated audio.

    Attributes:
        convs: Stack of Conv1d layers.
        conv_post: Final Conv1d layer.
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        """Initialise DiscriminatorS.

        Args:
            use_spectral_norm: If ``True`` use spectral norm; otherwise weight
                norm.
        """
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Discriminate the input waveform at full scale.

        Args:
            x: Waveform tensor of shape ``[B, 1, T]``.

        Returns:
            Tuple ``(score, fmap)`` where ``score`` is the flattened logit
            tensor of shape ``[B, T']`` and ``fmap`` is a list of intermediate
            feature maps.
        """
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# ---------------------------------------------------------------------------
# MultiPeriodDiscriminator
# ---------------------------------------------------------------------------


class MultiPeriodDiscriminator(torch.nn.Module):
    """Multi-period discriminator combining one scale and five period discriminators.

    Runs a :class:`DiscriminatorS` and five :class:`DiscriminatorP` instances
    (periods 2, 3, 5, 7, 11) on both real and generated audio and returns the
    discriminator outputs and feature maps for all sub-discriminators.

    Attributes:
        discriminators: ModuleList containing DiscriminatorS and five
            DiscriminatorP instances.
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        """Initialise MultiPeriodDiscriminator.

        Args:
            use_spectral_norm: If ``True`` use spectral norm in all sub-
                discriminators; otherwise weight norm.
        """
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Discriminate real vs. generated waveforms across all sub-discriminators.

        Args:
            y: Real waveform tensor of shape ``[B, 1, T]``.
            y_hat: Generated waveform tensor of shape ``[B, 1, T]``.

        Returns:
            Tuple ``(y_d_rs, y_d_gs, fmap_rs, fmap_gs)`` where each element
            is a list (one entry per sub-discriminator) of:
            - ``y_d_rs`` / ``y_d_gs``: discriminator logits for real / fake.
            - ``fmap_rs`` / ``fmap_gs``: feature maps for real / fake.
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ---------------------------------------------------------------------------
# ReferenceEncoder
# ---------------------------------------------------------------------------


class ReferenceEncoder(nn.Module):
    """Reference-audio encoder for speaker conditioning without a speaker ID.

    Extracts a fixed-length speaker embedding from a reference mel-spectrogram
    using a stack of strided Conv2d layers followed by a GRU.

    Inputs : ``[N, Ty/r, n_mels*r]``  (mel-spectrograms)
    Outputs: ``[N, ref_enc_gru_size]``

    Attributes:
        spec_channels: Number of mel-spectrogram channels.
        convs: ModuleList of weight-normed Conv2d layers.
        gru: GRU that aggregates spatial features over time.
        proj: Linear projection from GRU hidden size to ``gin_channels``.
        layernorm: Optional LayerNorm applied to the input (or ``None``).
    """

    def __init__(
        self,
        spec_channels: int,
        gin_channels: int = 0,
        layernorm: bool = False,
    ) -> None:
        """Initialise ReferenceEncoder.

        Args:
            spec_channels: Number of mel-spectrogram frequency bins.
            gin_channels: Output embedding dimension (speaker embedding size).
            layernorm: If ``True`` apply ``nn.LayerNorm`` to the input before
                the convolutional stack.
        """
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
            print('[Ref Enc]: using layer norm')
        else:
            self.layernorm = None

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a reference mel-spectrogram into a speaker embedding.

        Args:
            inputs: Mel-spectrogram tensor of shape ``[N, T, spec_channels]``.
            mask: Unused; kept for API compatibility.

        Returns:
            Speaker embedding tensor of shape ``[N, gin_channels]``.
        """
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(
        self,
        L: int,
        kernel_size: int,
        stride: int,
        pad: int,
        n_convs: int,
    ) -> int:
        """Compute the spatial size after ``n_convs`` strided convolutions.

        Args:
            L: Input spatial dimension.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            pad: Convolution padding.
            n_convs: Number of strided convolutions.

        Returns:
            Output spatial dimension after all convolutions.
        """
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


# ---------------------------------------------------------------------------
# SynthesizerTrn
# ---------------------------------------------------------------------------


class SynthesizerTrn(nn.Module):
    """End-to-end MeloTTS synthesiser for training and inference.

    Combines a TextEncoder, PosteriorEncoder, Generator,
    normalising-flow (ResidualCouplingBlock or TransformerCouplingBlock),
    StochasticDurationPredictor, and DurationPredictor.  Supports both
    speaker-ID conditioning and reference-audio conditioning.

    Attributes:
        n_vocab: Phoneme vocabulary size.
        spec_channels: Mel-spectrogram channel count.
        inter_channels: Intermediate latent dimension.
        hidden_channels: Hidden channel count used throughout the model.
        filter_channels: Feed-forward filter channels for the Transformer.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        kernel_size: Convolution kernel size.
        p_dropout: Dropout probability.
        resblock: Residual block variant string (``"1"`` or ``"2"``).
        resblock_kernel_sizes: Kernel sizes for the generator residual blocks.
        resblock_dilation_sizes: Dilation sizes for the generator residual blocks.
        upsample_rates: Upsampling strides for the generator.
        upsample_initial_channel: Initial channel count for upsampling.
        upsample_kernel_sizes: Kernel sizes for the generator upsampling layers.
        segment_size: Training segment size (frames).
        n_speakers: Number of speakers (0 = reference encoder mode).
        gin_channels: Global conditioning channel size.
        use_sdp: If ``True`` use stochastic duration predictor during inference.
        enc_p: TextEncoder.
        dec: Generator (waveform decoder).
        enc_q: PosteriorEncoder.
        flow: Normalising flow (TransformerCouplingBlock or
            ResidualCouplingBlock).
        sdp: StochasticDurationPredictor.
        dp: DurationPredictor.
        emb_g: Speaker embedding table (present when ``n_speakers > 0``).
        ref_enc: ReferenceEncoder (present when ``n_speakers == 0``).
        use_vc: If ``True`` disable text-encoder speaker conditioning.
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        n_speakers: int = 256,
        gin_channels: int = 256,
        use_sdp: bool = True,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 6,
        flow_share_parameter: bool = False,
        use_transformer_flow: bool = True,
        use_vc: bool = False,
        num_languages: Optional[int] = None,
        num_tones: Optional[int] = None,
        norm_refenc: bool = False,
        **kwargs,
    ) -> None:
        """Initialise SynthesizerTrn.

        Args:
            n_vocab: Phoneme vocabulary size.
            spec_channels: Number of mel-spectrogram channels (frequency bins).
            segment_size: Training segment length in frames.
            inter_channels: Intermediate latent dimension.
            hidden_channels: Hidden channel count.
            filter_channels: Transformer feed-forward filter channels.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer layers in the text encoder.
            kernel_size: Convolution kernel size.
            p_dropout: Dropout probability.
            resblock: Residual block variant: ``"1"`` or ``"2"``.
            resblock_kernel_sizes: Kernel sizes for the generator resblocks.
            resblock_dilation_sizes: Dilation sizes for the generator resblocks.
            upsample_rates: Strides for the generator upsampling stages.
            upsample_initial_channel: Generator upsampling initial channels.
            upsample_kernel_sizes: Kernel sizes for generator upsampling.
            n_speakers: Number of speaker IDs (0 = reference-encoder mode).
            gin_channels: Speaker embedding dimension.
            use_sdp: Use stochastic duration predictor during inference.
            n_flow_layer: Number of flow steps in the coupling block.
            n_layers_trans_flow: Number of Transformer layers per flow step.
            flow_share_parameter: Share Transformer weights across flow steps.
            use_transformer_flow: Use TransformerCouplingBlock; otherwise
                ResidualCouplingBlock.
            use_vc: Voice-conversion mode (disables text-encoder conditioning).
            num_languages: Number of language IDs (None → import from text).
            num_tones: Number of tone IDs (None → import from text).
            norm_refenc: Apply LayerNorm inside the ReferenceEncoder.
            **kwargs: Additional options:
                ``use_spk_conditioned_encoder`` (bool, default True),
                ``use_noise_scaled_mas`` (bool, default False),
                ``mas_noise_scale_initial`` (float, default 0.01),
                ``noise_scale_delta`` (float, default 2e-6).
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
            num_languages=num_languages,
            num_tones=num_tones,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        self.sdp = StochasticDurationPredictor(
            in_channels=hidden_channels,
            filter_channels=192,
            kernel_size=3,
            p_dropout=0.5,
            n_flows=4,
            gin_channels=gin_channels,
        )
        self.dp = DurationPredictor(
            in_channels=hidden_channels,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=gin_channels,
        )

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels, layernorm=norm_refenc)
        self.use_vc = use_vc

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        ja_bert: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        """Training forward pass.

        Encodes text and spectrogram, computes monotonic alignment, duration
        losses, and decodes a waveform segment.

        Args:
            x: Phoneme ID tensor of shape ``[B, T_text]``.
            x_lengths: Text sequence lengths of shape ``[B]``.
            y: Mel-spectrogram tensor of shape ``[B, spec_channels, T_spec]``.
            y_lengths: Spectrogram lengths of shape ``[B]``.
            sid: Speaker ID tensor of shape ``[B]``.
            tone: Tone ID tensor of shape ``[B, T_text]``.
            language: Language ID tensor of shape ``[B, T_text]``.
            bert: BERT feature tensor of shape ``[B, BERT_DIM, T_text]``.
            ja_bert: Japanese BERT feature tensor of shape
               ``[B, JA_BERT_DIM, T_text]``.

        Returns:
            Tuple of:
            - ``o``: Decoded waveform segment.
            - ``l_length``: Duration loss scalar.
            - ``attn``: Monotonic attention map.
            - ``ids_slice``: Segment slice indices.
            - ``x_mask``: Text mask.
            - ``y_mask``: Spectrogram mask.
            - ``(z, z_p, m_p, logs_p, m_q, logs_q)``: Latent variables.
            - ``(x, logw, logw_)``: Encoder output and log-duration predictions.
        """
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        if self.use_vc:
            g_p = None
        else:
            g_p = g
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)

        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging

        l_length = l_length_dp + l_length_sdp

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
        )

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        ja_bert: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
        sdp_ratio: float = 0,
        y: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Inference forward pass — synthesise a waveform from text.

        Args:
            x: Phoneme ID tensor of shape ``[B, T_text]``.
            x_lengths: Text sequence lengths of shape ``[B]``.
            sid: Speaker ID tensor of shape ``[B]``.
            tone: Tone ID tensor of shape ``[B, T_text]``.
            language: Language ID tensor of shape ``[B, T_text]``.
            bert: BERT feature tensor of shape ``[B, BERT_DIM, T_text]``.
            ja_bert: Japanese BERT feature tensor of shape
               ``[B, JA_BERT_DIM, T_text]``.
            noise_scale: Scale for the prior noise (controls naturalness).
            length_scale: Multiplier applied to predicted durations.
            noise_scale_w: Noise scale for the stochastic duration predictor.
            max_len: Optional maximum output length in frames.
            sdp_ratio: Mixing ratio between SDP (1.0) and DP (0.0) durations.
            y: Optional reference spectrogram (for reference-encoder mode).
            g: Optional pre-computed speaker embedding.

        Returns:
            Tuple ``(o, attn, y_mask, (z, z_p, m_p, logs_p))`` where ``o``
            is the synthesised waveform.
        """
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        if g is None:
            if self.n_speakers > 0:
                g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            else:
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        if self.use_vc:
            g_p = None
        else:
            g_p = g
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # print('max/min of o:', o.max(), o.min())
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid_src: torch.Tensor,
        sid_tgt: torch.Tensor,
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Convert speech from one speaker to another.

        Encodes the source audio with the source speaker embedding, passes
        through the flow, then decodes with the target speaker embedding.

        Args:
            y: Source mel-spectrogram tensor of shape
               ``[B, spec_channels, T]``.
            y_lengths: Source spectrogram lengths of shape ``[B]``.
            sid_src: Source speaker embedding tensor of shape
               ``[B, gin_channels, 1]``.
            sid_tgt: Target speaker embedding tensor of shape
               ``[B, gin_channels, 1]``.
            tau: Posterior sampling temperature (default 1.0).

        Returns:
            Tuple ``(o_hat, y_mask, (z, z_p, z_hat))`` where ``o_hat`` is
            the converted waveform.
        """
        g_src = sid_src
        g_tgt = sid_tgt
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, tau=tau)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
