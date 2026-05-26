"""Command-line entry point for MeloTTS single-file synthesis.

Exposes the ``melo`` CLI command (registered via ``pyproject.toml`` /
``setup.cfg`` console scripts) for synthesising a text string or file to a
WAV output path.

Usage::

    melo "Hello world" output.wav --language EN --speaker EN-Default --speed 1.0
"""

from __future__ import annotations

# stdlib
import os
import warnings

# third-party
import click

# local
from melo.api import TTS


@click.command
@click.argument('text')
@click.argument('output_path')
@click.option("--file", '-f', is_flag=True, show_default=True, default=False, help="Text is a file")
@click.option('--language', '-l', default='EN', help='Language, defaults to English', type=click.Choice(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], case_sensitive=False))
@click.option('--speaker', '-spk', default='EN-Default', help='Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Default"', type=click.Choice(['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU']))
@click.option('--speed', '-s', default=1.0, help='Speed, defaults to 1.0', type=float)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
def main(
    text: str,
    file: bool,
    output_path: str,
    language: str,
    speaker: str,
    speed: float,
    device: str,
) -> None:
    """Synthesise TEXT and write audio to OUTPUT_PATH.

    TEXT may be a literal string or, when --file/-f is provided, a path to a
    plain-text file whose contents are used as the input.

    Args:
        text: Input text string or path to a text file (when *file* is True).
        file: When ``True``, *text* is treated as a file path.
        output_path: Destination WAV file path.
        language: Language code (case-insensitive).
        speaker: Speaker ID string.  Only meaningful for English; ignored for
            other languages.
        speed: Speaking rate multiplier (1.0 = normal speed).
        device: PyTorch device string, e.g. ``'cpu'``, ``'cuda'``, ``'auto'``.
    """
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text) as f:
                text = f.read().strip()
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    language = language.upper()
    if language == '': language = 'EN'
    if speaker == '': speaker = None
    if (not language == 'EN') and speaker:
        warnings.warn('You specified a speaker but the language is English.')
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if language == 'EN':
        if not speaker: speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    model.tts_to_file(text=text, speaker_id=spkr, output_path=output_path, speed=speed)


if __name__ == "__main__":
    main()
