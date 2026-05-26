"""Text cleaning and sequence conversion utilities for MeloTTS.

This module provides high-level helpers that normalise raw input text, apply
grapheme-to-phoneme (G2P) conversion, and optionally extract BERT features —
all dispatched through a language-keyed module map so that adding a new
language requires only a single entry in ``language_module_map``.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from . import chinese, japanese, english, chinese_mix, korean, french, spanish
from . import cleaned_text_to_sequence


# Maps language code strings to their corresponding language-processing modules.
# Each module is expected to expose at minimum:
#   - text_normalize(text: str) -> str
#   - g2p(norm_text: str) -> (phones, tones, word2ph)
#   - get_bert_feature(norm_text, word2ph, device) -> torch.Tensor
# Note: both 'SP' and 'ES' map to the Spanish module.
language_module_map = {
    "ZH": chinese,
    "JP": japanese,
    "EN": english,
    "ZH_MIX_EN": chinese_mix,
    "KR": korean,
    "FR": french,
    "SP": spanish,
    "ES": spanish,
}


def clean_text(
    text: str,
    language: str,
) -> tuple[str, list[str], list[int], list[int]]:
    """Normalise text and run grapheme-to-phoneme conversion.

    Args:
        text: Raw input text string.
        language: Language code (e.g. ``"ZH"``, ``"EN"``).

    Returns:
        A 4-tuple ``(norm_text, phones, tones, word2ph)`` where:

        * **norm_text** – language-normalised text.
        * **phones** – list of phoneme symbol strings.
        * **tones** – list of tone integers, one per phoneme.
        * **word2ph** – list mapping each word to its phoneme count.
    """
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(
    text: str,
    language: str,
    device: Optional[str] = None,
) -> tuple[str, list[str], list[int], list[int], torch.Tensor]:
    """Normalise text, run G2P, and extract BERT features.

    The ``word2ph`` counts are doubled internally (with an extra +1 for the
    first element) before being passed to the BERT feature extractor to account
    for the ``[CLS]``/``[SEP]`` tokens.  The original ``word2ph`` is returned
    unmodified.

    Args:
        text: Raw input text string.
        language: Language code (e.g. ``"ZH"``, ``"EN"``).
        device: Target device string for the BERT model (e.g. ``"cuda:0"``).
            Passed directly to the language module's ``get_bert_feature``.

    Returns:
        A 5-tuple ``(norm_text, phones, tones, word2ph, bert)`` where:

        * **norm_text** – language-normalised text.
        * **phones** – list of phoneme symbol strings.
        * **tones** – list of tone integers, one per phoneme.
        * **word2ph** – original (unmodified) word-to-phoneme count list.
        * **bert** – BERT feature tensor with shape ``(hidden_size, num_phones)``.
    """
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(
    text: str,
    language: str,
) -> tuple[list[int], list[int], list[int]]:
    """Clean text and convert the resulting phones to integer ID sequences.

    Convenience wrapper that chains :func:`clean_text` and
    :func:`cleaned_text_to_sequence`.

    Args:
        text: Raw input text string.
        language: Language code (e.g. ``"ZH"``, ``"EN"``).

    Returns:
        A 3-tuple ``(phones, tones, lang_ids)`` — see
        :func:`~melo.text.cleaned_text_to_sequence` for details.
    """
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass