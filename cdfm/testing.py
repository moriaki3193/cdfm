# -*- coding: utf-8 -*-
"""Utilities for testing.
"""
import numpy as np
from .data import CDFMRow
from .consts import LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES


def is_equal_rows(this: CDFMRow, that: CDFMRow) -> bool:
    """Compare two given rows, and return if they are equal.
    """
    for i, field in enumerate(CDFMRow._fields):
        if field in {LABEL, QID, EID, CIDS}:
            if this[i] != that[i]:
                return False
        elif field in {FEATURES}:
            if not np.allclose(this[i], that[i]):
                return False
        elif field in {PROXIMITIES}:
            if (this[i] is None) or (that[i] is None):
                if not (this[i] is None) and (that[i] is None):
                    return False
            elif len(this[i]) != len(that[i]):
                return False
            else:
                for this_prox, that_prox in zip(this[i], that[i]):
                    if not np.allclose(this_prox, that_prox):
                        return False
    return True
