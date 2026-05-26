"""Spanish to IPA conversion module.

This module provides a utility function to convert Spanish text into
International Phonetic Alphabet (IPA) representation using Gruut.
"""

from typing import List, Union

from .cleaner import spanish_cleaners
from .gruut_wrapper import Gruut


def es2ipa(text: str) -> str:
    """Converts Spanish text to its IPA phoneme representation.

    Args:
        text (str): The input Spanish text to be converted.

    Returns:
        str: The phonemized text in IPA format.
    """
    e = Gruut(
        language="es-es",
        keep_puncs=True,
        keep_stress=True,
        use_espeak_phonemes=True,
    )
    # text = spanish_cleaners(text)
    phonemes: str = e.phonemize(text, separator="")
    return phonemes


if __name__ == "__main__":
    print(es2ipa("¿Y a quién echaría de menos, en el mundo si no fuese a vos?"))