# -*- coding: utf-8 -*-
"""Differentiation of model equations.
"""
from typing import List
import numpy as np
from ..config import DTYPE


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

def p_Iec_u(eind: int,
            cinds: int,
            Ve: np.ndarray,
            Vc: np.ndarray,
            d: np.ndarray,
            ps: np.ndarray) -> np.ndarray:
    """Partially differentiate p_Iec with u.

    Parameters:
        eind: an index of the target entity.
        cinds: indices of competitors.
        Ve: a matrix of latent vectors of entities. shape (p, k).
        Vc: a matrix of latent vectors of competitors. shape (p, k).
        d: a proximity factors. shape (|cinds|, #factors).
        ps: a probabilities of interaction occurrence.

    Returns:
        res: shape (#factors, ).
    """
    partial_sigmoid = np.einsum('i,ij->ij', np.multiply(ps, (DTYPE(1.) - ps)), -d)
    strengths = np.dot(Vc[cinds], Ve[eind])
    return np.einsum('ij,i->j', partial_sigmoid, strengths)

def p_Iec_ve(cinds: List[int], Vc: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Partially differentiate P_Iec with ve.

    Parameters:
        cinds: indices of competitors.
        Vc: a matrix of latent vectors of competitors. shape (p, k).
        ps: a probabilities of interaction occurrence.

    Returns:
        res: shape (k, ).
    """
    return np.einsum('ij,i->j', Vc.take(cinds, axis=0), ps)

def p_Iec_vc(eind: int, Ve: np.ndarray, p: DTYPE) -> np.ndarray:
    """Partially differentiate P_Iec with ve.

    Parameters:
        eind: an index of the target entity.
        Ve: a matrix of latent vectors of entities. shape (p, k).
        p: a probability of interaction occurrence.

    Returns:
        res: shape (k, ).
    """
    return np.multiply(p, Ve[eind])

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
