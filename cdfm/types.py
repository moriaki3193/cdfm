# -*- coding: utf-8 -*-
"""Type annotations.
"""
from typing import Dict, List, Union
from .data import CDFMRow


CDFMDataset = List[CDFMRow]
EntityID = Union[int, str]
EntIndMap = Dict[EntityID, int]
