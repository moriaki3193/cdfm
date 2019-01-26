# -*- coding: utf-8 -*-
"""Definition of records in dataset.
"""
from collections import namedtuple
from .consts import LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES


CDFMRow = namedtuple('CDFMRow', (LABEL, QID, EID, CIDS, FEATURES, PROXIMITIES))
_label_loc: int = CDFMRow._fields.index(LABEL)
_eid_loc: int = CDFMRow._fields.index(EID)
_cids_loc: int = CDFMRow._fields.index(CIDS)
_feat_loc: int = CDFMRow._fields.index(FEATURES)
_prox_loc: int = CDFMRow._fields.index(PROXIMITIES)
