"""MeloTTS — Multi-lingual Text-to-Speech library.

Package layout::

    melo/
    ├── nn/              Neural network building blocks (models, modules, losses…)
    ├── audio/           Audio / DSP utilities (mel-spectrogram, STFT…)
    ├── text/            Text normalisation and G2P, one sub-package per language
    │   ├── zh/          Chinese (core, bert, mix, tone_sandhi)
    │   ├── en/          English (core, bert, utils/)
    │   ├── fr/          French  (core, bert, phonemizer/)
    │   ├── es/          Spanish (core, bert, phonemizer/)
    │   ├── jp/          Japanese (core, bert)
    │   └── kr/          Korean  (core, ko_dictionary)
    ├── training/        Training loop and dataset utilities
    ├── cli/             CLI entry points
    │   ├── main.py      ``melo`` / ``melotts`` command  (text → WAV)
    │   └── infer.py     ``melo-infer`` command (checkpoint → WAV per speaker)
    ├── ui/
    │   └── app.py       ``melo-ui`` command  (Gradio WebUI)
    ├── utils/           Config schema, checkpoint helpers, audio loaders…
    ├── scripts/         Shell / Python helper scripts (train.sh, preprocess_text.py…)
    └── api.py           High-level TTS inference API

Quick start::

    from melo.api import TTS
    tts = TTS(language="EN", device="auto")
    tts.tts_to_file("Hello world", speaker_id=0, output_path="output.wav")

Backward-compatible re-exports so that existing code using the old flat
package layout continues to work::

    from melo.models import SynthesizerTrn   # old path still works
    from melo.commons import slice_segments  # old path still works
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# Keeps ``from melo.models import SynthesizerTrn`` working.
# ---------------------------------------------------------------------------
from melo.nn import commons  # noqa: F401
from melo.nn.models import (  # noqa: F401
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from melo.audio.mel_processing import (  # noqa: F401
    mel_spectrogram_torch,
    spectrogram_torch,
)
