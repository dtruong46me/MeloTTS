"""Module for English text processing and grapheme-to-phoneme conversion.

This module provides utilities for normalizing English text, converting words
to phonemes, and extracting BERT features for text-to-speech synthesis.
"""

import os
import pickle
import re
from typing import Dict, List, Tuple, Optional

import torch
from g2p_en import G2p
from transformers import AutoTokenizer

from . import symbols
from .english_utils.abbreviations import expand_abbreviations
from .english_utils.number_norm import normalize_numbers
from .english_utils.time_norm import expand_time_english
from .japanese import distribute_phone

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")
_g2p = G2p()

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def post_replace_ph(ph: str) -> str:
    """Replace a specific phoneme or punctuation mark with its standard equivalent.

    Args:
        ph (str): The input phoneme or punctuation mark.

    Returns:
        str: The processed phoneme.
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
        "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def read_dict() -> Dict[str, List[List[str]]]:
    """Read the CMU pronunciation dictionary from the file.

    Returns:
        Dict[str, List[List[str]]]: A dictionary mapping words to their syllable pronunciations.
    """
    g2p_dict: Dict[str, List[List[str]]] = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict: Dict[str, List[List[str]]], file_path: str) -> None:
    """Cache the parsed CMU dictionary to a pickle file.

    Args:
        g2p_dict (Dict[str, List[List[str]]]): The parsed pronunciation dictionary.
        file_path (str): The path to the cache file.
    """
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict() -> Dict[str, List[List[str]]]:
    """Get the CMU dictionary, loading from cache if available.

    Returns:
        Dict[str, List[List[str]]]: The pronunciation dictionary.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


eng_dict = get_dict()


def refine_ph(phn: str) -> Tuple[str, int]:
    """Extract the base phoneme and its tone from a phoneme string.

    Args:
        phn (str): The phoneme string (optionally ending with a digit).

    Returns:
        Tuple[str, int]: A tuple containing the lowercased base phoneme and its tone.
    """
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables: List[List[str]]) -> Tuple[List[str], List[int]]:
    """Flatten and refine a list of syllables into phonemes and tones.

    Args:
        syllables (List[List[str]]): A list of syllables, where each syllable is a list of phonemes.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing a list of base phonemes and a list of tones.
    """
    tones: List[int] = []
    phonemes: List[str] = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def text_normalize(text: str) -> str:
    """Normalize English text by expanding times, numbers, and abbreviations.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    text = text.lower()
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p_old(text: str) -> Tuple[List[str], List[int], List[int]]:
    """Convert text to phonemes using the old logic.

    Args:
        text (str): The input text.

    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing a list of phonemes,
            a list of tones, and a list indicating word-to-phoneme mapping.
    """
    tokenized = tokenizer.tokenize(text)
    # import pdb; pdb.set_trace()
    phones: List[str] = []
    tones: List[int] = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    # todo: implement word2ph
    word2ph = [1 for i in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph


def g2p(
    text: str, pad_start_end: bool = True, tokenized: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[int]]:
    """Convert text to phonemes with padding and word-to-phoneme mapping.

    Args:
        text (str): The input text.
        pad_start_end (bool, optional): Whether to pad the start and end of the phonemes. Defaults to True.
        tokenized (Optional[List[str]], optional): An optional list of tokens. Defaults to None.

    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing a list of phonemes,
            a list of tones, and a list indicating word-to-phoneme mapping.
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
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
            phone_len += len(phns)
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
                phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = [post_replace_ph(i) for i in phones]

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def get_bert_feature(
    text: str, word2ph: List[int], device: Optional[str] = None
) -> torch.Tensor:
    """Extract BERT features for the given text based on word-to-phone mapping.

    Args:
        text (str): The input English text.
        word2ph (List[int]): A list indicating the number of phones each word maps to.
        device (Optional[str], optional): The device to run the model on. Defaults to None.

    Returns:
        torch.Tensor: The phone-level BERT feature tensor.
    """
    from text import english_bert

    return english_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    from text.english_bert import get_bert_feature

    text = "In this paper, we propose 1 DSPGAN, a N-F-T GAN-based universal vocoder."
    text = text_normalize(text)
    phones, tones, word2ph = g2p(text)
    import pdb

    pdb.set_trace()
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)

    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
