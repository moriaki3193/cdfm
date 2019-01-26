# -*- coding: utf-8 -*-
import numpy as np
from .config import DTYPE


def compute_RMSE(pred_labels: np.ndarray, obs_labels: np.ndarray) -> DTYPE:
    """Root Mean Squared Error.
    """
    return np.sqrt(np.mean(np.square(pred_labels - obs_labels))).astype(DTYPE)
