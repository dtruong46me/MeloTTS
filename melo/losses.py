"""Loss functions for MeloTTS GAN-based vocoder training.

This module provides the loss functions used to train the generator and
discriminator components of MeloTTS, including:

- **Feature matching loss** (``feature_loss``): L1 distance between intermediate
  discriminator feature maps of real and generated audio.
- **Discriminator loss** (``discriminator_loss``): Least-squares GAN loss for the
  discriminator, encouraging real outputs toward 1 and generated outputs toward 0.
- **Generator loss** (``generator_loss``): Least-squares GAN loss for the generator,
  encouraging generated outputs toward 1.
- **KL divergence loss** (``kl_loss``): KL divergence between the posterior and prior
  distributions in the VITS-style latent variable model.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def feature_loss(
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Compute the feature matching loss between real and generated feature maps.

    For each discriminator and each of its internal feature maps, computes the
    mean absolute error between the real (detached) and generated activations,
    then sums across all layers and discriminators. The result is scaled by 2.

    Args:
        fmap_r: Nested list of feature map tensors from the discriminator applied
            to real audio. Shape: ``[num_discriminators][num_layers][B, C, T]``.
        fmap_g: Nested list of feature map tensors from the discriminator applied
            to generated audio. Same structure as ``fmap_r``.

    Returns:
        Scalar tensor representing the total feature matching loss.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Compute the least-squares GAN discriminator loss.

    Encourages the discriminator to output values close to 1 for real audio
    and close to 0 for generated audio using the LSGAN formulation.

    Args:
        disc_real_outputs: List of discriminator output tensors for real audio,
            one tensor per sub-discriminator.
        disc_generated_outputs: List of discriminator output tensors for generated
            audio, one tensor per sub-discriminator.

    Returns:
        A tuple ``(loss, r_losses, g_losses)`` where:
        - ``loss``: Scalar tensor — the total discriminator loss summed over all
          sub-discriminators.
        - ``r_losses``: List of per-discriminator real loss values (Python floats).
        - ``g_losses``: List of per-discriminator generated loss values (Python floats).
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute the least-squares GAN generator loss.

    Encourages the generator to produce audio that the discriminator rates
    close to 1 (i.e., indistinguishable from real audio).

    Args:
        disc_outputs: List of discriminator output tensors for generated audio,
            one tensor per sub-discriminator.

    Returns:
        A tuple ``(loss, gen_losses)`` where:
        - ``loss``: Scalar tensor — the total generator loss summed over all
          sub-discriminators.
        - ``gen_losses``: List of per-discriminator generator loss tensors.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked KL divergence loss between prior and posterior.

    Measures how much the posterior distribution ``q`` deviates from the prior
    distribution ``p``, averaged over valid (non-padded) time steps.

    Args:
        z_p: Latent samples from the prior, shape ``[B, H, T_t]``.
        logs_q: Log standard deviation of the posterior ``q``, shape ``[B, H, T_t]``.
        m_p: Mean of the prior ``p``, shape ``[B, H, T_t]``.
        logs_p: Log standard deviation of the prior ``p``, shape ``[B, H, T_t]``.
        z_mask: Binary mask indicating valid time steps, shape ``[B, 1, T_t]`` or
            broadcastable to the KL tensor.

    Returns:
        Scalar tensor — the KL divergence averaged over all valid (masked) positions.
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
