# -*- coding: utf-8 -*-
"""Model equation components.
"""
from typing import List
import numpy as np
from ..config import DTYPE


def Iec(eind: int, cinds: List[int], Ve: np.ndarray, Vc: np.ndarray) -> DTYPE:
    """Strength of interaction between Entity and Competitors.

    Parameters:
        eind: an index of the target entity.
        cinds: indices of competitors.
        Ve: a matrix of latent vectors of entities. shape (p, k).
        Vc: a matrix of latent vectors of competitors. shape (p, k).

    Returns:
        res: strength of interaction between entity and competitors.
    """
    return np.sum(np.dot(Vc[cinds], Ve[eind]), axis=0)

def p_Iec(eind: int, cinds: List[int], Ve: np.ndarray, Vc: np.ndarray, ps: np.ndarray) -> DTYPE:
    """Expected strength of interaction between Entity and Competitors.

    Parameters:
        eind: an index of the target entity.
        cinds: indices of competitors.
        Ve: a matrix of latent vectors of entities. shape (p, k).
        Vc: a matrix of latent vectors of competitors. shape (p, k).
        ps: probabiities of interaction occurrence between 2 entities.

    Returns:
        res: strength of interaction between entity and competitors.
    """
    interactions = np.dot(Vc[cinds], Ve[eind])
    return np.sum(np.multiply(ps, interactions), axis=0)

def Ief(eind: int, x: np.ndarray, Ve: np.ndarray, Vf: np.ndarray) -> DTYPE:
    """Strength of interaction between Entity and its Features.

    Parameters:
        eind: an index of the target entity.
        x: a feature vector of the target entity.
        Ve: a matrix of latent vectors of entities. shape (p, k).
        Vf: a matrix of latent vectors of features. shape (q, k).

    Returns:
        res: strength of interaction between entity and its features.
    """
    return np.dot(np.dot(Vf, Ve[eind]), x)

def Iff(x: np.ndarray, Vf: np.ndarray) -> DTYPE:
    """Strength of interaction between 2 features.

    Parameters:
        x: a feature vector.
        Vf: a matrix of latent vectors of features. shape(q, k).

    Returns:
        res: strength of interaction between 2 features.
    """
    Vfx = np.multiply(Vf.T, x).T
    squared_sum = np.square(np.sum(Vfx, axis=0))
    sum_squared = np.sum(np.square(Vfx), axis=0)
    coef = DTYPE(.5)
    return coef * np.sum(squared_sum - sum_squared)
