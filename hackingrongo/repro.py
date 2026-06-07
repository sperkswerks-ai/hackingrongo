"""
hackingrongo.repro — Global reproducibility utilities.

Call set_global_seed() once at the start of any entry point (pipeline.py
main(), every Ring 1 script, every Ring 1 module __main__ block) to make
every stochastic operation in that process deterministic.
"""

from __future__ import annotations

import logging
import random

log = logging.getLogger(__name__)

DEFAULT_SEED: int = 20260606


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python random, NumPy, and PyTorch (if installed).

    PyTorch: also sets cudnn.deterministic=True and cudnn.benchmark=False
    so convolution algorithms are chosen deterministically even on GPU.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    log.info("Global seed set: %d", seed)
