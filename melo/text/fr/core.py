"""Module for French text processing and phonemization for MeloTTS."""

import pickle
import os
import re
from typing import List, Tuple, Any, Optional, Dict

from . import symbols
from .fr_phonemizer import cleaner as fr_cleaner
from .fr_phonemizer import fr_to_ipa
from transformers import AutoTokenizer


def distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """Distributes phonemes evenly across words.

    Args:
        n_phone (int): The total number of phonemes.
        n_word (int): The total number of words.

    Returns:
        List[int]: A list of integers representing the number of phonemes for each word.
    """
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def text_normalize(text: str) -> str:
    """Normalizes French text by applying cleaners.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """
    text = fr_cleaner.french_cleaners(text)
    return text


model_id = "dbmdz/bert-base-french-europeana-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p(text: str, pad_start_end: bool = True, tokenized: Optional[List[str]] = None) -> Tuple[List[str], List[int], List[int]]:
    """Converts graphemes to phonemes (G2P) for French text.

    Args:
        text (str): The text to process.
        pad_start_end (bool, optional): Whether to pad the start and end of the output. Defaults to True.
        tokenized (Optional[List[str]], optional): Pre-tokenized text. Defaults to None.

    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing:
            - A list of phonemes.
            - A list of tones.
            - A list indicating the number of phonemes per word.
    """
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    # import pdb; pdb.set_trace()
    phs: List[str] = []
    ph_groups: List[List[str]] = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))

    phones: List[str] = []
    tones: List[int] = []
    word2ph: List[int] = []
    # print(ph_groups)
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w == "[UNK]":
            phone_list = ["UNK"]
        else:
            phone_list = list(filter(lambda p: p != " ", fr_to_ipa.fr2ipa(w)))

        for ph in phone_list:
            phones.append(ph)
            tones.append(0)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
        # print(phone_list, aaa)
        # print('=' * 10)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def get_bert_feature(text: str, word2ph: List[int], device: Optional[Any] = None) -> Any:
    """Retrieves BERT features for the provided text.

    Args:
        text (str): The input text.
        word2ph (List[int]): The mapping of words to phonemes.
        device (Optional[Any], optional): The device to use for computation. Defaults to None.

    Returns:
        Any: The BERT features corresponding to the input text.
    """
    from text import french_bert
    return french_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    ori_text = "Ce service gratuit est“”\"\" 【disponible》 en chinois 【simplifié] et autres 123"
    # ori_text = "Ils essayaient vainement de faire comprendre à ma mère qu'avec les cent mille francs que m'avait laissé mon père,"
    # print(ori_text)
    text = text_normalize(ori_text)
    print(text)
    phoneme = fr_to_ipa.fr2ipa(text)
    print(phoneme)

    from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer
    from text.cleaner_multiling import unicleaners

    def _test_text_normalize(text: str) -> str:
        text = unicleaners(text, cased=True, lang="fr")
        return text

    # print(ori_text)
    text = _test_text_normalize(ori_text)
    print(text)
    phonemizer = MultiPhonemizer({"fr-fr": "espeak"})
    # phonemizer.lang_to_phonemizer['fr'].keep_stress = True
    # phonemizer.lang_to_phonemizer['fr'].use_espeak_phonemes = True
    phoneme = phonemizer.phonemize(text, separator="", language="fr-fr")
    print(phoneme)