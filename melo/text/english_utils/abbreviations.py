"""Module for expanding common English abbreviations.

This module provides functionality to convert common English abbreviations
into their full spoken forms using regular expressions.
"""

import re

# List of (regular expression, replacement) pairs for abbreviations in english:
abbreviations_en = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str, lang: str = "en") -> str:
    """Expand abbreviations in the given text.

    Args:
        text (str): The input text containing abbreviations.
        lang (str, optional): The language code. Defaults to "en".

    Returns:
        str: The text with abbreviations expanded to their full forms.

    Raises:
        NotImplementedError: If a language other than "en" is provided.
    """
    if lang == "en":
        _abbreviations = abbreviations_en
    else:
        raise NotImplementedError()
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text