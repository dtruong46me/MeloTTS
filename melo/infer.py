"""Command-line inference script for MeloTTS.

Loads a TTS model from a local checkpoint and synthesises speech for every
speaker defined in the model configuration, writing each result to a separate
WAV file under *output_dir*.

Usage::

    python -m melo.infer -m /path/to/G_latest.pth -t "Hello world" -l EN -o outputs
"""

from __future__ import annotations

# stdlib
import os

# third-party
import click

# local
from melo.api import TTS


@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")
def main(ckpt_path: str | None, text: str | None, language: str, output_dir: str) -> None:
    """Synthesise *text* with every speaker found in the checkpoint config.

    Args:
        ckpt_path: Path to the model checkpoint file (required).  The
            corresponding ``config.json`` must reside in the same directory.
        text: Text string to synthesise.
        language: Language code matching the checkpoint (e.g. ``'EN'``).
        output_dir: Root directory for output WAV files.  A sub-directory is
            created for each speaker.
    """
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")

    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)

    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = f'{output_dir}/{spk_name}/output.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text=text, speaker_id=spk_id, output_path=save_path)


if __name__ == "__main__":
    main()
