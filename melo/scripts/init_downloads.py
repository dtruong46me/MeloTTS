"""Pre-download all MeloTTS language models.

Run this script once to eagerly download and cache the model checkpoints and
configuration files for all supported languages so that subsequent runs do not
require a network connection.

Usage::

    python -m melo.init_downloads
"""

from __future__ import annotations

# local
from melo.api import TTS


def download_all_models(device: str = 'auto') -> dict[str, TTS]:
    """Instantiate a TTS model for every supported language.

    Loading each :class:`~melo.api.TTS` instance triggers the automatic
    download of the corresponding config and checkpoint from Hugging Face Hub
    (if not already cached locally).

    Args:
        device: PyTorch device string passed to each :class:`~melo.api.TTS`
            constructor.  Use ``'auto'`` (default) to select the best
            available device automatically.

    Returns:
        A dictionary mapping language codes (``'EN'``, ``'ES'``, ``'FR'``,
        ``'ZH'``, ``'JP'``, ``'KR'``) to their loaded :class:`~melo.api.TTS`
        instances.
    """
    models = {
        'EN': TTS(language='EN', device=device),
        'ES': TTS(language='ES', device=device),
        'FR': TTS(language='FR', device=device),
        'ZH': TTS(language='ZH', device=device),
        'JP': TTS(language='JP', device=device),
        'KR': TTS(language='KR', device=device),
    }
    return models


if __name__ == '__main__':
    download_all_models(device='auto')