"""Set of default text cleaners.

Provides functions to clean, normalize, and format text strings
prior to phonemization for different languages.
"""

import re
from typing import Dict

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

rep_map: Dict[str, str] = {
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
    """Replaces certain punctuation marks based on a predefined map.

    Args:
        text (str): The input text.

    Returns:
        str: The text with specific punctuation replaced.
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def lowercase(text: str) -> str:
    """Converts the text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The lowercased text.
    """
    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Collapses consecutive whitespaces into a single space and strips boundaries.

    Args:
        text (str): The input text.

    Returns:
        str: The text with collapsed whitespaces.
    """
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text: str) -> str:
    """Removes standard punctuation marks from the beginning of the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text without leading punctuation.
    """
    return re.sub(r"^[,.!?]+", "", text)


def remove_aux_symbols(text: str) -> str:
    """Removes auxiliary symbols such as brackets, quotes, and guillemets.

    Args:
        text (str): The input text.

    Returns:
        str: The text without auxiliary symbols.
    """
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»\']+", "", text)
    return text


def replace_symbols(text: str, lang: str = "en") -> str:
    """Replaces symbols based on the language tag.

    Args:
        text (str): Input text.
        lang (str, optional): Language identifier. e.g., "en", "fr", "pt", "ca", "es".
            Defaults to "en".

    Returns:
        str: The modified text. For example, if text="si l'avi cau, diguem-ho"
        and lang="ca", returns "si lavi cau, diguemho".
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


def spanish_cleaners(text: str) -> str:
    """Basic pipeline for Spanish text.

    Cleans Spanish text by applying lowercase, symbol replacement, punctuation
    replacement, and auxiliary symbol removal. Adds a period at the end if missing.

    Args:
        text (str): The input Spanish text.

    Returns:
        str: The cleaned Spanish text.
    """
    text = lowercase(text)
    text = replace_symbols(text, lang="es")
    text = replace_punctuation(text)
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r"([^\.,!\?\-…])$", r"\1.", text)
    return text
