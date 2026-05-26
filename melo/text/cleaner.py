"""Text cleaning and sequence conversion utilities for MeloTTS.

This module provides high-level helpers that normalise raw input text, apply
grapheme-to-phoneme (G2P) conversion, and optionally extract BERT features —
all dispatched through a language-keyed module map so that adding a new
language requires only a single entry in language_module_map.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from .zh import core as chinese
from .jp import core as japanese
from .en import core as english
from .zh import mix as chinese_mix
from .kr import core as korean
from .fr import core as french
from .es import core as spanish
from .. import cleaned_text_to_sequence


# Maps language code strings to their corresponding language-processing modules.
# Each module is expected to expose at minimum:
#   - text_normalize(text: str) -> str
#   - g2p(norm_text: str) -> tuple[list[str], list[int], list[int]]
#   - get_bert_feature(norm_text: str, word2ph: list[int], device: str) -> torch.Tensor
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


def clean_text(text: str, language: str) -> tuple[str, list[str], list[int], list[int]]:
    """Normalise text and run grapheme-to-phoneme conversion.

    Args:
        text (str): Raw input text string.
        language (str): Language code (e.g., "ZH", "EN").

    Returns:
        tuple[str, list[str], list[int], list[int]]: A 4-tuple containing:
            - norm_text (str): Language-normalised text.
            - phones (list[str]): List of phoneme symbol strings.
            - tones (list[int]): List of tone integers, one per phoneme.
            - word2ph (list[int]): List mapping each word to its phoneme count.
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

    The word2ph counts are doubled internally (with an extra +1 for the
    first element) before being passed to the BERT feature extractor to account
    for the [CLS]/[SEP] tokens. The original word2ph is returned unmodified.

    Args:
        text (str): Raw input text string.
        language (str): Language code (e.g., "ZH", "EN").
        device (Optional[str], optional): Target device string for the BERT model (e.g., "cuda:0").
            Passed directly to the language module's get_bert_feature. Defaults to None.

    Returns:
        tuple[str, list[str], list[int], list[int], torch.Tensor]: A 5-tuple containing:
            - norm_text (str): Language-normalised text.
            - phones (list[str]): List of phoneme symbol strings.
            - tones (list[int]): List of tone integers, one per phoneme.
            - word2ph (list[int]): Original (unmodified) word-to-phoneme count list.
            - bert (torch.Tensor): BERT feature tensor with shape (hidden_size, num_phones).
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


def text_to_sequence(text: str, language: str) -> tuple[list[int], list[int], list[int]]:
    """Clean text and convert the resulting phones to integer ID sequences.

    Convenience wrapper that chains clean_text and cleaned_text_to_sequence.

    Args:
        text (str): Raw input text string.
        language (str): Language code (e.g., "ZH", "EN").

    Returns:
        tuple[list[int], list[int], list[int]]: A 3-tuple containing:
            - phones (list[int]): List of integer phone IDs.
            - tones (list[int]): List of integer tone IDs.
            - lang_ids (list[int]): List of language IDs, one per phone.
    """
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass