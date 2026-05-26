"""Training pipeline for MeloTTS.

Sub-package grouping all training-specific code:

* :mod:`melo.training.data_utils` — dataset, collate function, and distributed
  bucket sampler used by the training loop.
* :mod:`melo.training.train` — distributed training loop (run via
  ``torchrun`` / ``python -m melo.training.train``).

Typical usage::

    # Launch training with torchrun
    torchrun --nproc_per_node=4 -m melo.training.train --c configs/config.json
"""

from __future__ import annotations
