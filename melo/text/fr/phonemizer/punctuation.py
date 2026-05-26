"""Module for handling punctuations in text processing."""

import collections
import re
from enum import Enum
from typing import List, Tuple, Any

import six

_DEF_PUNCS: str = ';:,.!?¡¿—…"«»“”'

_PUNC_IDX = collections.namedtuple("_punc_index", ["punc", "position"])


class PuncPosition(Enum):
    """Enum for the punctuation positions."""

    BEGIN = 0
    END = 1
    MIDDLE = 2
    ALONE = 3


class Punctuation:
    """Handle punctuations in text.

    Just strip punctuations from text or strip and restore them later.

    Args:
        puncs (str): The punctuations to be processed. Defaults to `_DEF_PUNCS`.

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
        """Initializes the Punctuation instance.

        Args:
            puncs (str): The punctuations to be processed. Defaults to _DEF_PUNCS.
        """
        self.puncs = puncs

    @staticmethod
    def default_puncs() -> str:
        """Return default set of punctuations.

        Returns:
            str: The default punctuations string.
        """
        return _DEF_PUNCS

    @property
    def puncs(self) -> str:
        """str: The string of punctuation characters."""
        return self._puncs

    @puncs.setter
    def puncs(self, value: str) -> None:
        if not isinstance(value, six.string_types):
            raise ValueError("[!] Punctuations must be of type str.")
        self._puncs = "".join(list(dict.fromkeys(list(value))))  # remove duplicates without changing the order
        self.puncs_regular_exp = re.compile(rf"(\s*[{re.escape(self._puncs)}]+\s*)+")

    def strip(self, text: str) -> str:
        """Remove all the punctuations by replacing with `space`.

        Args:
            text (str): The text to be processed.

        Returns:
            str: The text without punctuations.

        Example::
            "This is. example !" -> "This is example "
        """
        return re.sub(self.puncs_regular_exp, " ", text).rstrip().lstrip()

    def strip_to_restore(self, text: str) -> Tuple[List[str], List[Any]]:
        """Remove punctuations from text to restore them later.

        Args:
            text (str): The text to be processed.

        Returns:
            Tuple[List[str], List[Any]]: A tuple containing the split text and the punctuation map.

        Examples ::
            "This is. example !" -> (["This is", "example"], [".", "!"])
        """
        text_split, puncs = self._strip_to_restore(text)
        return text_split, puncs

    def _strip_to_restore(self, text: str) -> Tuple[List[str], List[Any]]:
        """Auxiliary method for strip_to_restore to do the actual processing.

        Args:
            text (str): The text to be processed.

        Returns:
            Tuple[List[str], List[Any]]: The split text and list of punctuation metadata.
        """
        matches = list(re.finditer(self.puncs_regular_exp, text))
        if not matches:
            return [text], []
        # the text is only punctuations
        if len(matches) == 1 and matches[0].group() == text:
            return [], [_PUNC_IDX(text, PuncPosition.ALONE)]
        # build a punctuation map to be used later to restore punctuations
        puncs = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match == matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match == matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PUNC_IDX(match.group(), position))
        # convert str text to a List[str], each item is separated by a punctuation
        splitted_text = []
        for idx, punc in enumerate(puncs):
            split = text.split(punc.punc)
            prefix, suffix = split[0], punc.punc.join(split[1:])
            splitted_text.append(prefix)
            # if the text does not end with a punctuation, add it to the last item
            if idx == len(puncs) - 1 and len(suffix) > 0:
                splitted_text.append(suffix)
            text = suffix
        return splitted_text, puncs

    @classmethod
    def restore(cls, text: List[str], puncs: List[Any]) -> List[str]:
        """Restore punctuation in a text.

        Args:
            text (List[str]): The text parts to be processed.
            puncs (List[Any]): The list of punctuations map to be used for restoring.

        Returns:
            List[str]: The restored text.

        Examples ::
            ['This is', 'example'], ['.', '!'] -> ["This is. example!"]
        """
        return cls._restore(text, puncs, 0)

    @classmethod
    def _restore(cls, text: List[str], puncs: List[Any], num: int) -> List[str]:  # pylint: disable=too-many-return-statements
        """Auxiliary recursive method for restore.

        Args:
            text (List[str]): The text parts.
            puncs (List[Any]): The punctuation metadata map.
            num (int): Current recursion depth / index.

        Returns:
            List[str]: The restored text items.
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
            return [current.punc] + cls._restore(text, puncs[1:], num + 1)

        # POSITION == MIDDLE
        if len(text) == 1:  # pragma: nocover
            # a corner case where the final part of an intermediate
            # mark (I) has not been phonemized
            return cls._restore([text[0] + current.punc], puncs[1:], num)

        return cls._restore([text[0] + current.punc + text[1]] + text[2:], puncs[1:], num)