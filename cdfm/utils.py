# -*- coding: utf-8 -*-
"""Utilities module
"""
from typing import Any, Iterable, Set
from operator import itemgetter
from .types import CDFMDataset, EntityID, EntIndMap
from .consts import EID


def _extract_first(items: Iterable[Any]) -> Any:
    first_item = itemgetter(0)
    return first_item(items)

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
