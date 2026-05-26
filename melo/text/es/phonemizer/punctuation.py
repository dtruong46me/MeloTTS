"""Punctuation handling for phonemization.

This module provides tools to strip punctuations from text and
optionally restore them after processing.
"""

import collections
import re
from enum import Enum
from typing import Any, List, Tuple, Union

import six

_DEF_PUNCS = ';:,.!?¡¿—…"«»“”'

_PUNC_IDX = collections.namedtuple("_punc_index", ["punc", "position"])


class PuncPosition(Enum):
    """Enum for the punctuations positions."""

    BEGIN = 0
    END = 1
    MIDDLE = 2
    ALONE = 3


class Punctuation:
    """Handle punctuations in text.

    Just strip punctuations from text or strip and restore them later.

    Args:
        puncs (str, optional): The punctuations to be processed. Defaults to `_DEF_PUNCS`.

    Example:
        >>> punc = Punctuation()
        >>> punc.strip("This is. example !")
        'This is example'

        >>> text_striped, punc_map = punc.strip_to_restore("This is. example !")
        >>> ' '.join(text_striped)
        'This is example'

        >>> text_restored = punc.restore(text_striped, punc_map)
        >>> text_restored[0]
        'This is. example !'
    """

    def __init__(self, puncs: str = _DEF_PUNCS) -> None:
        """Initializes the Punctuation handler."""
        self.puncs = puncs

    @staticmethod
    def default_puncs() -> str:
        """Return default set of punctuations.

        Returns:
            str: A string containing default punctuation characters.
        """
        return _DEF_PUNCS

    @property
    def puncs(self) -> str:
        """Gets the punctuation characters.

        Returns:
            str: The string of punctuation characters.
        """
        return self._puncs

    @puncs.setter
    def puncs(self, value: Union[str, Any]) -> None:
        """Sets the punctuation characters and compiles the regex pattern.

        Args:
            value (Union[str, Any]): A string of punctuations to use.

        Raises:
            ValueError: If the provided value is not a string.
        """
        if not isinstance(value, six.string_types):
            raise ValueError("[!] Punctuations must be of type str.")
        self._puncs = "".join(list(dict.fromkeys(list(value))))  # remove duplicates without changing the order
        self.puncs_regular_exp = re.compile(rf"(\s*[{re.escape(self._puncs)}]+\s*)+")

    def strip(self, text: str) -> str:
        """Remove all the punctuations by replacing with `space`.

        Args:
            text (str): The text to be processed.

        Returns:
            str: The text with punctuations removed and stripped of leading/trailing spaces.

        Example:
            "This is. example !" -> "This is example"
        """
        return re.sub(self.puncs_regular_exp, " ", text).rstrip().lstrip()

    def strip_to_restore(self, text: str) -> Tuple[List[str], List[Any]]:
        """Remove punctuations from text to restore them later.

        Args:
            text (str): The text to be processed.

        Returns:
            Tuple[List[str], List[Any]]: A tuple containing a list of text segments
            and a list of punctuation map objects.

        Examples:
            "This is. example !" -> (["This is", "example"], [...])
        """
        text_segments, puncs = self._strip_to_restore(text)
        return text_segments, puncs

    def _strip_to_restore(self, text: str) -> Tuple[List[str], List[Any]]:
        """Auxiliary method to strip and track punctuations.

        Args:
            text (str): The text to be processed.

        Returns:
            Tuple[List[str], List[Any]]: The split text segments and the tracked punctuations.
        """
        matches = list(re.finditer(self.puncs_regular_exp, text))
        if not matches:
            return [text], []
        # the text is only punctuations
        if len(matches) == 1 and matches[0].group() == text:
            return [], [_PUNC_IDX(text, PuncPosition.ALONE)]
        # build a punctuation map to be used later to restore punctuations
        puncs: List[Any] = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match == matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match == matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PUNC_IDX(match.group(), position))
        # convert str text to a List[str], each item is separated by a punctuation
        splitted_text: List[str] = []
        for idx, punc in enumerate(puncs):
            split = text.split(punc.punc)
            prefix, suffix = split[0], punc.punc.join(split[1:])
            splitted_text.append(prefix)
            # if the text does not end with a punctuation, add it to the last item
            if idx == len(puncs) - 1 and len(suffix) > 0:
                splitted_text.append(suffix)
            text = suffix
        while splitted_text and splitted_text[0] == "":
            splitted_text = splitted_text[1:]
        return splitted_text, puncs

    @classmethod
    def restore(cls, text: List[str], puncs: List[Any]) -> List[str]:
        """Restore punctuation in a text.

        Args:
            text (List[str]): The text segments to be processed.
            puncs (List[Any]): The list of punctuations map to be used for restoring.

        Returns:
            List[str]: The text segments with punctuations restored.

        Examples:
            ['This is', 'example'], [...] -> ["This is. example!"]
        """
        return cls._restore(text, puncs, 0)

    @classmethod
    def _restore(cls, text: List[str], puncs: List[Any], num: int) -> List[str]:  # pylint: disable=too-many-return-statements
        """Auxiliary recursive method to restore punctuations.

        Args:
            text (List[str]): The text segments.
            puncs (List[Any]): The punctuation objects.
            num (int): Recursion depth counter.

        Returns:
            List[str]: The reconstructed text segments.
        """
        if not puncs:
            return text

        # nothing have been phonemized, returns the puncs alone
        if not text:
            return ["".join(m.punc for m in puncs)]

        current = puncs[0]

        if current.position == PuncPosition.BEGIN:
            return cls._restore([current.punc + text[0]] + text[1:], puncs[1:], num)

        if current.position == PuncPosition.END:
            return [text[0] + current.punc] + cls._restore(text[1:], puncs[1:], num + 1)

        if current.position == PuncPosition.ALONE:
            return [current.mark] + cls._restore(text, puncs[1:], num + 1)  # type: ignore

        # POSITION == MIDDLE
        if len(text) == 1:  # pragma: nocover
            # a corner case where the final part of an intermediate
            # mark (I) has not been phonemized
            return cls._restore([text[0] + current.punc], puncs[1:], num)

        return cls._restore([text[0] + current.punc + text[1]] + text[2:], puncs[1:], num)


# if __name__ == "__main__":
#     punc = Punctuation()
#     text = "This is. This is, example!"

#     print(punc.strip(text))

#     split_text, puncs = punc.strip_to_restore(text)
#     print(split_text, " ---- ", puncs)

#     restored_text = punc.restore(split_text, puncs)
#     print(restored_text)