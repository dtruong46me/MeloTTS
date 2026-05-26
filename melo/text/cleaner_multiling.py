"""Set of default text cleaners.

This module provides basic text cleaning pipelines and utilities for different languages,
handling punctuation replacement, lowercase conversion, and symbol removal.
"""

import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

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
    "...": ".",
    "…": ".",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text: str) -> str:
    """Replace specific punctuation marks in the text based on a predefined map.

    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation replaced.
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def lowercase(text: str) -> str:
    """Convert the input text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The lowercased text.
    """
    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space and strip edges.

    Args:
        text (str): The input text.

    Returns:
        str: The text with collapsed whitespace.
    """
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text: str) -> str:
    """Remove any punctuation marks at the beginning of the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without leading punctuation.
    """
    return re.sub(r"^[,.!?]+", "", text)


def remove_aux_symbols(text: str) -> str:
    """Remove auxiliary symbols such as brackets and quotes from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without auxiliary symbols.
    """
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»\']+", "", text)
    return text


def replace_symbols(text: str, lang: str = "en") -> str:
    """Replace symbols based on the language tag.

    Args:
        text (str): Input text.
        lang (str, optional): Language identifier (e.g., "en", "fr", "pt", "ca"). Defaults to "en".

    Returns:
        str: The modified text. For example, for input args text: "si l'avi cau, diguem-ho"
            and lang: "ca", the output will be "si lavi cau, diguemho".
    """
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
    elif lang == "es":
        text = text.replace("&", "y")
        text = text.replace("'", "")
    return text


def unicleaners(text: str, cased: bool = False, lang: str = "en") -> str:
    """Apply a basic text cleaning pipeline.

    There is no need to expand abbreviations and numbers, as the phonemizer already does that.

    Args:
        text (str): The input text.
        cased (bool, optional): If False, convert text to lowercase. Defaults to False.
        lang (str, optional): Language identifier. Defaults to "en".

    Returns:
        str: The cleaned text.
    """
    if not cased:
        text = lowercase(text)
    text = replace_punctuation(text)
    text = replace_symbols(text, lang=lang)
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r"([^\.,!\?\-…])$", r"\1.", text)
    return text
