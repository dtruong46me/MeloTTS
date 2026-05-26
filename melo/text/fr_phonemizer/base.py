"""Module providing the base class for phonemizers."""

import abc
from typing import List, Tuple, Optional, Any, Dict

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
        language (str):
            Language used by the phonemizer.
        punctuations (str):
            List of punctuation marks to be preserved. Defaults to Punctuation.default_puncs().
        keep_puncs (bool):
            Whether to preserve punctuation marks or not. Defaults to False.
    """

    def __init__(self, language: str, punctuations: str = Punctuation.default_puncs(), keep_puncs: bool = False) -> None:
        """Initializes the base phonemizer.

        Args:
            language (str): Language used by the phonemizer.
            punctuations (str, optional): List of punctuation marks to be preserved.
            keep_puncs (bool, optional): Whether to preserve punctuation marks or not.
        """
        # ensure the backend is installed on the system
        if not self.is_available():
            raise RuntimeError("{} not installed on your system".format(self.name()))  # pragma: nocover

        # ensure the backend support the requested language
        self._language: str = self._init_language(language)

        # setup punctuation processing
        self._keep_puncs: bool = keep_puncs
        self._punctuator: Punctuation = Punctuation(punctuations)

    def _init_language(self, language: str) -> str:
        """Language initialization.

        This method may be overloaded in child classes (see Segments backend).

        Args:
            language (str): The language code to initialize.

        Returns:
            str: The initialized language code.
        """
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the {self.name()} backend')
        return language

    @property
    def language(self) -> str:
        """str: The language code configured to be used for phonemization."""
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
    def supported_languages() -> List[str]:
        """Return a list of language codes supported by the backend.

        Returns:
            List[str]: A list of supported language codes.
        """
        ...

    def is_supported_language(self, language: str) -> bool:
        """Returns True if `language` is supported by the backend.

        Args:
            language (str): The language to check.

        Returns:
            bool: True if supported, False otherwise.
        """
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        """The main phonemization method.

        Args:
            text (str): Text to phonemize.
            separator (str): Separator to use between phonemes.

        Returns:
            str: The phonemized text.
        """
        ...

    def _phonemize_preprocess(self, text: str) -> Tuple[List[str], List[Any]]:
        """Preprocess the text before phonemization.

        1. remove spaces
        2. remove punctuation

        Override this if you need a different behaviour.

        Args:
            text (str): The input text.

        Returns:
            Tuple[List[str], List[Any]]: Preprocessed text chunks and punctuation metadata.
        """
        text = text.strip()
        if self._keep_puncs:
            # a tuple (text, punctuation marks)
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: List[str], punctuations: List[Any]) -> str:
        """Postprocess the raw phonemized output.

        Override this if you need a different behaviour.

        Args:
            phonemized (List[str]): The phonemized text chunks.
            punctuations (List[Any]): The punctuation metadata.

        Returns:
            str: The postprocessed text.
        """
        if self._keep_puncs:
            return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: str, separator: str = "|", language: Optional[str] = None) -> str:  # pylint: disable=unused-argument
        """Returns the `text` phonemized for the given language.

        Args:
            text (str):
                Text to be phonemized.
            separator (str):
                string separator used between phonemes. Default to '|'.
            language (Optional[str]):
                Language to use, though often unused in the base class.

        Returns:
            str: Phonemized text.
        """
        text_split, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in text_split:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        phonemized_str: str = self._phonemize_postprocess(phonemized, punctuations)
        return phonemized_str

    def print_logs(self, level: int = 0) -> None:
        """Prints logs for the phonemizer.

        Args:
            level (int): Indentation level for the logs. Defaults to 0.
        """
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")