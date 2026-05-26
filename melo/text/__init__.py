"""Text processing package for MeloTTS.

This module exposes core utilities for converting cleaned phoneme sequences into
integer ID sequences and for retrieving language-specific BERT features.  It
re-exports all public names from :mod:`melo.text.symbols` so that callers can
access ``symbols``, ``language_tone_start_map``, ``language_id_map``, etc.
directly from this package.
"""

from __future__ import annotations

from typing import Optional

import torch

from .symbols import *


# Build a reverse mapping: symbol string -> integer ID, derived from the
# global ``symbols`` list that is imported via the wildcard above.
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(
    cleaned_text: list[str],
    tones: list[int],
    language: str,
    symbol_to_id: Optional[dict[str, int]] = None,
) -> tuple[list[int], list[int], list[int]]:
    """Convert a cleaned phoneme sequence to integer ID sequences.

    Translates each phoneme symbol into its corresponding integer ID, offsets
    tone values by the language-specific tone start index, and builds a
    per-phone language ID list.

    Args:
        cleaned_text: List of phoneme symbol strings produced by a G2P module.
        tones: List of raw tone integers, one per phoneme in ``cleaned_text``.
        language: Language code (e.g. ``"ZH"``, ``"EN"``), used to look up
            tone and language ID offsets in ``language_tone_start_map`` and
            ``language_id_map``.
        symbol_to_id: Optional custom symbol-to-ID mapping.  When ``None``
            (default) the module-level ``_symbol_to_id`` mapping is used.

    Returns:
        A 3-tuple ``(phones, tones, lang_ids)`` where:

        * **phones** – list of integer phone IDs.
        * **tones** – list of integer tone IDs, offset by the language tone
          start value.
        * **lang_ids** – list of language IDs, one per phone.
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(
    norm_text: str,
    word2ph: list[int],
    language: str,
    device: str | torch.device,
) -> torch.Tensor:
    """Retrieve language-specific BERT features for the given text.

    Lazily imports the BERT feature extractor for the requested language and
    invokes it with the provided text and word-to-phoneme mapping.

    Args:
        norm_text: Normalised input text string.
        word2ph: List mapping each word to its phoneme count.
        language: Language code selecting which BERT model to use.  Supported
            codes: ``"ZH"``, ``"EN"``, ``"JP"``, ``"ZH_MIX_EN"``, ``"FR"``,
            ``"SP"``, ``"ES"``, ``"KR"``.
        device: Target device for the BERT tensor (e.g. ``"cuda:0"`` or
            ``"cpu"``).

    Returns:
        A ``torch.Tensor`` of BERT features with shape
        ``(hidden_size, num_phones)``.
    """
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert
    from .chinese_mix import get_bert_feature as zh_mix_en_bert
    from .spanish_bert import get_bert_feature as sp_bert
    from .french_bert import get_bert_feature as fr_bert
    from .korean import get_bert_feature as kr_bert

    # Maps each supported language code to its BERT feature-extraction function.
    # Note: both 'SP' (Spanish shorthand) and 'ES' (ISO 639-1) share the same
    # Spanish BERT extractor.
    lang_bert_func_map = {
        "ZH": zh_bert,
        "EN": en_bert,
        "JP": jp_bert,
        "ZH_MIX_EN": zh_mix_en_bert,
        "FR": fr_bert,
        "SP": sp_bert,
        "ES": sp_bert,
        "KR": kr_bert,
    }
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
