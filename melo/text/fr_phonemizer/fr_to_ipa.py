"""Module for converting French text to International Phonetic Alphabet (IPA)."""

from .cleaner import french_cleaners
from .gruut_wrapper import Gruut


def remove_consecutive_t(input_str: str) -> str:
    """Removes consecutive 't' characters if there are three or more.

    Args:
        input_str (str): The input string to process.

    Returns:
        str: The processed string with consecutive 't's limited.
    """
    result = []
    count = 0

    for char in input_str:
        if char == "t":
            count += 1
        else:
            if count < 3:
                result.extend(["t"] * count)
            count = 0
            result.append(char)

    if count < 3:
        result.extend(["t"] * count)

    return "".join(result)


def fr2ipa(text: str) -> str:
    """Converts a French text string to its IPA representation.

    Args:
        text (str): The French text to convert.

    Returns:
        str: The IPA representation of the input text.
    """
    e = Gruut(language="fr-fr", keep_puncs=True, keep_stress=True, use_espeak_phonemes=True)
    # text = french_cleaners(text)
    phonemes = e.phonemize(text, separator="")
    # print(phonemes)
    phonemes = remove_consecutive_t(phonemes)
    # print(phonemes)
    return phonemes