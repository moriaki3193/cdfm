# -*- coding: utf-8 -*-
"""Definition of records in dataset.
"""
from collections import namedtuple
from .consts import LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES


Row = namedtuple('Row', (LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES))
