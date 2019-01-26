# -*- coding: utf-8 -*-
"""Utilities module
"""
import os
from typing import Any, Callable, Iterable, List, Set
from operator import itemgetter
import numpy as np
import pandas as pd
from .config import DTYPE
from .consts import LABEL, QID, EID, CID, FEATURES, PROXIMITIES
from .types import CDFMDataset, EntityID, EntIndMap


def _extract_first(items: Iterable[Any]) -> Any:
    first_item = itemgetter(0)
    return first_item(items)

def load_cdfmdata(path: str,
                  ndim: int,
                  mode: str = 'feature',
                  tokenizer: str = ' ',
                  splitter: str = ':',
                  comment_symb: str = '#',
                  zero_indexed: bool = False) -> pd.DataFrame:
    """Load dataset from a given path.

    Parameters:
        path: path to a file.
        mode: one of ( 'feature' | 'proximity' ).
        tokenizer: symbol for decomposing a into key-value pair.
        splitter: symbol for splitting a key-value pair.
        comment_symb: symbol representing the begining of inline comments.
        zero_indexed: whether the feature vectors 0-indexed or not.

    Features line format:
        label qid:123 eid:abc 1:1.5 2:0.2 4:-0.9 ... # comment

    Proximities line format:
        qid:123 eid:abc cid:xyz 1:0.5 2:-0.3 ... # comment

    Returns:
        data: DataFrame object of the loaded dataset.
    """
    # Validation on a given path.
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found.')

    # Validation on a specified mode.
    if mode not in ['feature', 'proximity']:
        raise ValueError('mode should be either "feature" or "proximity".')

    # Features parser
    def _parse_features(line: str) -> tuple:
        line = line.rstrip()
        elems = _extract_first(line.split(comment_symb))
        tokens = elems.rstrip().split(tokenizer)
        label = float(tokens[0])
        _, qid = tokens[1].split(splitter)
        _, eid = tokens[2].split(splitter)
        features = np.zeros(ndim, dtype=DTYPE)
        for kv in tokens[3:]:
            nth_dim, raw_val = kv.split(splitter)
            dim_idx = int(nth_dim) if zero_indexed else int(nth_dim) - 1
            features[dim_idx] = DTYPE(raw_val)
        return (label, qid, eid, features)

    # Proximities parser
    def _parse_proximities(line: str) -> tuple:
        line = line.rstrip()
        elems = _extract_first(line.split(comment_symb))
        tokens = elems.rstrip().split(tokenizer)
        _, qid = tokens[0].split(splitter)
        _, eid = tokens[1].split(splitter)
        _, cid = tokens[2].split(splitter)
        factors = np.zeros(ndim, dtype=DTYPE)
        for kv in tokens[3:]:
            nth_dim, raw_val = kv.split(splitter)
            dim_idx = int(nth_dim) if zero_indexed else int(nth_dim) - 1
            factors[dim_idx] = DTYPE(raw_val)
        return (qid, eid, cid, factors)

    # Select a parser and columns
    _parse: Callable[[str], tuple]
    _columns: tuple
    if mode == 'feature':
        _parse = _parse_features
        _columns = (LABEL, QID, EID, FEATURES)
    elif mode == 'proximity':
        _parse = _parse_proximities
        _columns = (QID, EID, CID, PROXIMITIES)

    # Extracted rows
    rows: List[tuple] = []
    append_to_rows = rows.append

    # Parse lines
    with open(path, mode='r') as fp:
        line = fp.readline()
        while line:
            row = _parse(line)
            append_to_rows(row)
            line = fp.readline()

    return pd.DataFrame(rows, columns=_columns)

def make_map(unique_ids: Iterable[EntityID]) -> EntIndMap:
    """f: entity_id -> entity_index

    Parameters:
        unique_ids: an iterable object whose elements are entity ids.

    Returns:
        ent_ind_map: entity-index mapper.
    """
    return {eid: ind for ind, eid in enumerate(unique_ids)}

def extract_unique_ids(dataset: CDFMDataset, col: Any = EID) -> Set[EntityID]:
    """Extract unique ids in a given column in dataset.

    Parameters:
        dataset: a list of instance rows.
        col: a target column name.

    Returns:
        unique_ids: a set of unique ids in the column.
    """
    first_record = _extract_first(dataset)
    loc_idx = first_record._fields.index(col)
    return {row[loc_idx] for row in dataset}
