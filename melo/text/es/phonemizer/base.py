"""Base module for phonemization.

This module defines the abstract base class for phonemizers, including common
preprocessing, phonemization, and postprocessing workflows.
"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

from .punctuation import Punctuation


class BasePhonemizer(abc.ABC):
    """Base phonemizer class.

    Phonemization follows the following steps:
        1. Preprocessing:
            - remove empty lines
            - remove punctuation
            - keep track of punctuation marks

        2. Phonemization:
            - convert text to phonemes

        3. Postprocessing:
            - join phonemes
            - restore punctuation marks

    Args:
        language (str): Language used by the phonemizer.
        punctuations (Union[str, List[str]], optional): List of punctuation marks to be preserved.
            Defaults to Punctuation.default_puncs().
        keep_puncs (bool, optional): Whether to preserve punctuation marks or not. Defaults to False.
    """

    def __init__(
        self,
        language: str,
        punctuations: Union[str, List[str]] = Punctuation.default_puncs(),
        keep_puncs: bool = False,
    ) -> None:
        """Initializes the base phonemizer."""
        # ensure the backend is installed on the system
        if not self.is_available():
            raise RuntimeError(f"{self.name()} not installed on your system")  # pragma: nocover

        # ensure the backend support the requested language
        self._language = self._init_language(language)

        # setup punctuation processing
        self._keep_puncs = keep_puncs
        self._punctuator = Punctuation(punctuations)

    def _init_language(self, language: str) -> str:
        """Language initialization.

        This method may be overloaded in child classes (see Segments backend).

        Args:
            language (str): The language code to initialize.

        Returns:
            str: The initialized language code.
        
        Raises:
            RuntimeError: If the language is not supported by the backend.
        """
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the {self.name()} backend')
        return language

    @property
    def language(self) -> str:
        """The language code configured to be used for phonemization.

        Returns:
            str: The configured language code.
        """
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """The name of the backend.

        Returns:
            str: The name of the backend.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Returns True if the backend is installed, False otherwise.

        Returns:
            bool: Availability of the backend.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def version(cls) -> str:
        """Return the backend version.

        Returns:
            str: The backend version.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages() -> Union[List[str], Dict[str, str]]:
        """Return a collection of supported languages.

        Returns:
            Union[List[str], Dict[str, str]]: The supported languages.
        """
        ...

    def is_supported_language(self, language: str) -> bool:
        """Returns True if `language` is supported by the backend.

        Args:
            language (str): The language code to check.

        Returns:
            bool: True if the language is supported, False otherwise.
        """
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        """The main phonemization method.

        Args:
            text (str): The text to phonemize.
            separator (str): The separator to use between phonemes.

        Returns:
            str: The phonemized text.
        """

    def _phonemize_preprocess(self, text: str) -> Tuple[List[str], List[Any]]:
        """Preprocesses the text before phonemization.

        1. remove spaces
        2. remove punctuation

        Override this if you need a different behaviour.

        Args:
            text (str): The raw text to preprocess.

        Returns:
            Tuple[List[str], List[Any]]: A tuple containing a list of text segments
            and a list of punctuation marks.
        """
        text = text.strip()
        if self._keep_puncs:
            # a tuple (text, punctuation marks)
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: List[str], punctuations: List[Any]) -> str:
        """Postprocesses the raw phonemized output.

        Override this if you need a different behaviour.

        Args:
            phonemized (List[str]): The list of phonemized segments.
            punctuations (List[Any]): The list of punctuation marks to restore.

        Returns:
            str: The final reconstructed phonemized string.
        """
        if self._keep_puncs:
            return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: str, separator: str = "|", language: Optional[str] = None) -> str:  # pylint: disable=unused-argument
        """Returns the `text` phonemized for the given language.

        Args:
            text (str): Text to be phonemized.
            separator (str, optional): string separator used between phonemes. Defaults to '|'.
            language (Optional[str], optional): Language code. Defaults to None.

        Returns:
            str: Phonemized text.
        """
        text_segments, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in text_segments:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        phonemized_result = self._phonemize_postprocess(phonemized, punctuations)
        return phonemized_result

    def print_logs(self, level: int = 0) -> None:
        """Prints logging information about the phonemizer.

        Args:
            level (int, optional): Indentation level. Defaults to 0.
        """
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")