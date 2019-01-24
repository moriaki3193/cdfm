# -*- coding: utf-8 -*-
"""Differentiation of model equations.
"""
from typing import List
import numpy as np


def Iec_ve(cinds: List[int], Vc: np.ndarray) -> np.ndarray:
    """Partially differentiate Iec with ve.

    Parameters:
        cinds: indices of competitors.
        Vc: a matrix of latent vectors of competitors. shape (p, k).

    Returns:
        res: shape (k, ).
    """
    return np.sum(Vc.take(cinds, axis=0), axis=0)

def Iec_vc(eind: int, Ve: np.ndarray) -> np.ndarray:
    """Partially differentiate Iec with vc.

    Parameters:
        eind: an index of the target entity.
        Ve: a matrix of latent vectors of entities. shape (p, k).

    Returns:
        res: shape (k, ).
    """
    return Ve[eind]

def Ief_ve(x: np.ndarray, Vf: np.ndarray) -> np.ndarray:
    """Partially differentiate Ief with ve.

    Parameters:
        x: a feature vector of the target entity.
        Vf: a matrix of latent vectors of features. shape (q, k).

    Returns:
        res: shape (k, ).
    """
    return np.sum(np.multiply(Vf.T, x).T, axis=0)

def Ief_vf(eind: int, x: np.ndarray, Ve: np.ndarray) -> np.ndarray:
    """Partially differentiate Ief with vf.

    Parameters:
        eind: an index of the target entity.
        x: a feature vector of the target entity.
        Ve: a matrix of latent vectors of entities. shape (p, k).

    Returns:
        res: shape (q, k).
    """
    return np.outer(x, Ve[eind])

def Iff_vf(x: np.ndarray, Vf: np.ndarray) -> np.ndarray:
    """Partially differentiate Iff with vf.

    Parameters:
        x: a feature vector of the target entity.
        Vf: a matrix of latent vectors of features. shape (q, k).

    Returns:
        res: shape (q, k).
    """
    summation = np.sum((Vf.T * x).T, axis=0)
    first_term = np.outer(x, summation)
    second_term = (Vf.T * np.square(x)).T
    return first_term - second_term
