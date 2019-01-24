# -*- coding: utf-8 -*-
"""Utilities module
"""
from typing import Iterable
from .types import EntityID, EntIndMap


def make_map(unique_ids: Iterable[EntityID]) -> EntIndMap:
    """f: entity_id -> entity_index

    Parameters:
        unique_ids: an iterable object whose elements are entity ids.

    Returns:
        ent_ind_map: entity-index mapper.
    """
    return {eid: ind for ind, eid in enumerate(unique_ids)}
