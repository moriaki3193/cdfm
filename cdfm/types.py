# -*- coding: utf-8 -*-
"""Type annotations.
"""
from typing import Dict, List, Union
from .data import Row


Dataset = List[Row]
EntityID = Union[int, str]
EntIndMap = Dict[EntityID, int]
