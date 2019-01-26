# -*- coding: utf-8 -*-
"""Definition of records in dataset.
"""
from collections import namedtuple
from .consts import LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES


CDFMRow = namedtuple('CDFMRow', (LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES))
