"""Gradio WebUI for MeloTTS.

WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS

Launches a Gradio interface that lets users select a language, speaker, and
speaking speed, enter text, and download the synthesised audio.

Note:
    Make sure you've downloaded unidic (python -m unidic download) for this
    WebUI to work.
"""

from __future__ import annotations

# stdlib
import io
import os

# third-party
import click
import gradio as gr
import torch

# local
from melo.api import TTS

print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")

# ---------------------------------------------------------------------------
# Global model loading – required at module level for Gradio to function
# ---------------------------------------------------------------------------

speed = 1.0
device = 'auto'

models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}

speaker_ids = models['EN'].hps.data.spk2id

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',
}


def synthesize(
    speaker: str,
    text: str,
    speed: float,
    language: str,
    progress: gr.Progress = gr.Progress(),
) -> bytes:
    """Synthesise speech and return raw WAV bytes.

    Args:
        speaker: Speaker name key from ``hps.data.spk2id``.
        text: Text to synthesise.
        speed: Speaking rate multiplier.
        language: Language code (e.g. ``'EN'``, ``'ZH'``).
        progress: Gradio progress tracker used to display a progress bar.

    Returns:
        Raw WAV audio bytes suitable for a Gradio ``Audio`` component.
    """
    bio = io.BytesIO()
    models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
    return bio.getvalue()


def load_speakers(language: str, text: str) -> tuple[gr.update, str]:
    """Update the speaker dropdown and default text when the language changes.

    Args:
        language: Newly selected language code.
        text: Current text in the textbox.

    Returns:
        A tuple of ``(gr.update, new_text)`` where ``gr.update`` refreshes
        the speaker dropdown choices and ``new_text`` is the updated textbox
        value.
    """
    if text in list(default_text_dict.values()):
        newtext = default_text_dict[language]
    else:
        newtext = text
    return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys())), newtext


# ---------------------------------------------------------------------------
# Gradio UI – must remain at module level so Gradio can register the blocks
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown('# MeloTTS WebUI\n\nA WebUI for MeloTTS.')
    with gr.Group():
        speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-US', label='Speaker')
        language = gr.Radio(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], label='Language', value='EN')
        speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="Text to speak", value=default_text_dict['EN'])
        language.input(load_speakers, inputs=[language, text], outputs=[speaker, text])
    btn = gr.Button('Synthesize', variant='primary')
    aud = gr.Audio(interactive=False)
    btn.click(synthesize, inputs=[speaker, text, speed, language], outputs=[aud])
    gr.Markdown('WebUI by [mrfakename](https://twitter.com/realmrfakename).')


@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
def main(share: bool, host: str | None, port: int | None) -> None:
    """Launch the MeloTTS Gradio WebUI.

    Args:
        share: When ``True``, Gradio creates a publicly-accessible tunnel URL.
        host: Hostname or IP address to bind the server to.
        port: TCP port to listen on.  ``None`` lets Gradio choose a free port.
    """
    demo.queue(api_open=False).launch(show_api=False, share=share, server_name=host, server_port=port)


if __name__ == "__main__":
    main()
