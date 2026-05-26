"""MeloTTS — Multi-lingual Text-to-Speech library.

Package layout::

    melo/
    ├── nn/          Neural network building blocks (models, modules, losses…)
    ├── audio/       Audio / DSP utilities (mel-spectrogram, STFT…)
    ├── text/        Text normalisation and G2P per language
    ├── training/    Training loop and dataset utilities
    ├── monotonic_align/  C extension for monotonic alignment
    ├── api.py       High-level TTS inference API
    ├── app.py       Gradio WebUI entry point
    ├── infer.py     CLI inference script
    ├── main.py      Main CLI entry point (``melo`` / ``melotts`` command)
    ├── split_utils.py   Sentence-splitting utilities
    ├── download_utils.py  Model download helpers
    └── utils.py     General utilities (checkpoint I/O, HParams, logging…)

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
