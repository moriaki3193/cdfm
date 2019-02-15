# -*- coding: utf-8 -*-
"""Type annotations.
"""
from typing import Dict, List, Union
import numpy as np
from .data import CDFMRow
from .config import DTYPE


# CDFM
CDFMDataset = List[CDFMRow]
EntityID = Union[int, str]
EntIndMap = Dict[EntityID, int]

# CDER
DocumentID = Union[int, str]
QueryID = Union[int, str]
Vector = Union[np.ndarray]
Label = Union[DTYPE, float, int]
