"""Spanish text processing and grapheme-to-phoneme (G2P) conversion.

This module provides functions for normalizing Spanish text, converting it to phonemes,
and extracting BERT features for Text-to-Speech (TTS) models.
"""

import os
import pickle
import re
from typing import Any, List, Optional, Tuple

from transformers import AutoTokenizer

from . import symbols
from .es_phonemizer import cleaner as es_cleaner
from .es_phonemizer import es_to_ipa

model_id = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """Distributes the total number of phonemes evenly across the number of words.

    Args:
        n_phone (int): The total number of phonemes.
        n_word (int): The total number of words.

    Returns:
        List[int]: A list of integers representing the number of phonemes per word.
    """
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def text_normalize(text: str) -> str:
    """Normalizes Spanish text using predefined cleaners.

    Args:
        text (str): The input Spanish text to normalize.

    Returns:
        str: The normalized Spanish text.
    """
    text = es_cleaner.spanish_cleaners(text)
    return text


def post_replace_ph(ph: str) -> str:
    """Replaces specific phonemes or symbols with standard equivalents.

    Args:
        ph (str): The input phoneme or symbol.

    Returns:
        str: The replaced or normalized phoneme. If it's an unknown symbol, returns 'UNK'.
    """
    rep_map = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
    }
    if ph in rep_map:
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def refine_ph(phn: str) -> Tuple[str, int]:
    """Refines a phoneme by separating the base phoneme from its tone.

    Args:
        phn (str): The input phoneme, potentially containing a trailing tone digit.

    Returns:
        Tuple[str, int]: A tuple containing the lowercase base phoneme and its integer tone.
    """
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables: List[List[str]]) -> Tuple[List[str], List[int]]:
    """Refines a list of syllables into flat lists of phonemes and tones.

    Args:
        syllables (List[List[str]]): A list of syllables, where each syllable is a list of phonemes.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing a flat list of refined phonemes
        and a flat list of their corresponding tones.
    """
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def g2p(
    text: str,
    pad_start_end: bool = True,
    tokenized: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[int]]:
    """Converts Spanish text to phonemes, tones, and word-to-phoneme alignments.

    Args:
        text (str): The input Spanish text.
        pad_start_end (bool, optional): Whether to pad the sequence with placeholders ('_'). Defaults to True.
        tokenized (Optional[List[str]], optional): An optional list of pre-tokenized words. Defaults to None.

    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing:
            - A list of phonemes.
            - A list of tones.
            - A list of word-to-phoneme alignment counts.
    """
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    
    ph_groups: List[List[str]] = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones: List[str] = []
    tones: List[int] = []
    word2ph: List[int] = []
    
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == '[UNK]':
            phone_list = ['UNK']
        else:
            phone_list = list(filter(lambda p: p != " ", es_to_ipa.es2ipa(w)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def get_bert_feature(text: str, word2ph: List[int], device: Optional[str] = None) -> Any:
    """Extracts BERT features for the given text aligned to phonemes.

    Args:
        text (str): The input Spanish text.
        word2ph (List[int]): The word-to-phoneme alignment counts.
        device (Optional[str], optional): The device to run the model on (e.g., 'cpu', 'cuda'). Defaults to None.

    Returns:
        Any: The extracted BERT features as a PyTorch tensor.
    """
    from text import spanish_bert
    return spanish_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    test_text = "en nuestros tiempos estos dos pueblos ilustres empiezan a curarse, gracias sólo a la sana y vigorosa higiene de 1789."
    test_text = text_normalize(test_text)
    print(test_text)
    test_phones, test_tones, test_word2ph = g2p(test_text)
    test_bert = get_bert_feature(test_text, test_word2ph)
    print(test_phones)
    print(len(test_phones), test_tones, sum(test_word2ph), test_bert.shape)
