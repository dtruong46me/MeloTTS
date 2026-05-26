"""Set of default text cleaners for French text processing."""
# TODO: pick the cleaner for languages dynamically

import re
from typing import Dict
from .french_abbreviations import abbreviations_fr

# Regular expression matching whitespace:
_whitespace_re: re.Pattern = re.compile(r"\s+")


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
    "“": "",
    "”": "",
    "‘": "",
    "’": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "",
    "」": "",
    "¿": "",
    "¡": ""
}


def replace_punctuation(text: str) -> str:
    """Replaces specific punctuation marks based on the predefined mapping.

    Args:
        text (str): The input text to process.

    Returns:
        str: The text with punctuation replaced.
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def expand_abbreviations(text: str, lang: str = "fr") -> str:
    """Expands abbreviations in the text based on the specified language.

    Args:
        text (str): The input text.
        lang (str): The language code (e.g., "fr"). Defaults to "fr".

    Returns:
        str: The text with abbreviations expanded.
    """
    _abbreviations = []
    if lang == "fr":
        _abbreviations = abbreviations_fr
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text: str) -> str:
    """Converts the text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The lowercased text.
    """
    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Collapses multiple whitespace characters into a single space and strips boundaries.

    Args:
        text (str): The input text.

    Returns:
        str: The text with collapsed whitespace.
    """
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text: str) -> str:
    """Removes punctuation marks from the beginning of the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with leading punctuation removed.
    """
    return re.sub(r"^[,.!?]+", "", text)


def remove_aux_symbols(text: str) -> str:
    """Removes auxiliary symbols like brackets and quotation marks.

    Args:
        text (str): The input text.

    Returns:
        str: The text with auxiliary symbols removed.
    """
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return text


def replace_symbols(text: str, lang: str = "en") -> str:
    """Replaces symbols based on the language tag.

    Args:
        text (str): Input text.
        lang (str): Language identifier. Ex: "en", "fr", "pt", "ca". Defaults to "en".

    Returns:
        str: The modified text.
        Example:
            input args:
                text: "si l'avi cau, diguem-ho"
                lang: "ca"
            Output:
                text: "si lavi cau, diguemho"
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


def french_cleaners(text: str) -> str:
    """Pipeline for cleaning French text.

    There is no need to expand numbers, phonemizer already does that.

    Args:
        text (str): The input French text.

    Returns:
        str: The cleaned text.
    """
    text = expand_abbreviations(text, lang="fr")
    # text = lowercase(text) # as we use the cased bert
    text = replace_punctuation(text)
    text = replace_symbols(text, lang="fr")
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r"([^\.,!\?\-…])$", r"\1.", text)
    return text
