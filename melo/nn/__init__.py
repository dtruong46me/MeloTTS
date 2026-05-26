"""Neural network building blocks for MeloTTS.

Sub-package containing all trainable PyTorch modules:

* :mod:`melo.nn.commons` — shared tensor utilities and weight initialisation.
* :mod:`melo.nn.attentions` — multi-head attention, Encoder, Decoder, FFN.
* :mod:`melo.nn.modules` — WaveNet layers, residual blocks, coupling layers.
* :mod:`melo.nn.transforms` — normalising-flow spline transforms.
* :mod:`melo.nn.losses` — adversarial, feature-matching, and KL losses.
* :mod:`melo.nn.models` — top-level model definitions (SynthesizerTrn, etc.).

Frequently used symbols are re-exported here for convenient access::

    from melo.nn import commons
    from melo.nn.models import SynthesizerTrn
    from melo.nn.losses import generator_loss, discriminator_loss
"""

from __future__ import annotations

# Re-export the commons sub-module so callers can do ``from melo.nn import commons``
from . import commons  # noqa: F401

# Re-export key model classes
from .models import (  # noqa: F401
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)

# Re-export loss functions
from .losses import (  # noqa: F401
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
