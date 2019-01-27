# -*- coding: utf-8 -*-
"""Metaclasses of models.

All of the models must be named like `<ModelName>Meta`.
For example, the metaclass of variants of CDFMRanker should be named
like `<CDFMRanker>Meta`.
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from ..config import DTYPE
from ..consts import EID
from ..data import _label_loc, _eid_loc, _cids_loc, _feat_loc, _prox_loc
from ..measures import compute_RMSE
from ..types import CDFMDataset, EntIndMap
from ..utils import _extract_first, extract_unique_ids, make_map


class CDFMRankerMeta(metaclass=ABCMeta):
    """Metaclass of Combination-dependent Entity Ranker.

    Attributes:
        k: #dimensions of latent vectors.
        l2_w: L2 regularization on pointwise weights.
        l2_V: L2 regularization on pairwise weights.
        n_iter: #iterations when fitting.
        init_eta: initial learning rate.
        init_scale: v ~ Normal(0., scale=init_scale).
        scores: training evaluation scores.
        map: a map from EntityID to Index.

    Methods:
        _init_params: parameter initialization in the model.
        fit: model fitting using SGD algorithm.
        predict: make prediction on a given dataset.
    """

    _label_loc: int = _label_loc
    _eid_loc: int = _eid_loc
    _cids_loc: int = _cids_loc
    _feat_loc: int = _feat_loc
    _prox_loc: int = _prox_loc

    @staticmethod
    def _evaluate_score(pred_labels: np.ndarray, obs_labels) -> DTYPE:
        return compute_RMSE(pred_labels, obs_labels)

    def __init__(self,
                 k: int = 10,
                 l2_w: float = 1e-2,
                 l2_V: float = 1e-2,
                 n_iter: int = 1000,
                 init_eta: float = 1e-1,
                 init_scale: float = 1e-2) -> None:
        self.k = k
        self.l2_w = DTYPE(l2_w)
        self.l2_V = DTYPE(l2_V)
        self.n_iter = n_iter
        self.eta = DTYPE(init_eta)
        self.init_scale = init_scale
        self.scores = np.zeros(n_iter, dtype=DTYPE)
        # Dynamically set variables
        self.map: EntIndMap
        self.b: DTYPE
        self.w: np.ndarray
        self.Ve: np.ndarray
        self.Vc: np.ndarray
        self.Vf: np.ndarray

    def _init_params(self, data: CDFMDataset) -> None:
        first_row = _extract_first(data)
        n_features: int = len(first_row[self._feat_loc])
        unique_ids = extract_unique_ids(data, col=EID)
        n_uniques = len(unique_ids)
        self.map = make_map(unique_ids)
        self.b = DTYPE(0.)
        self.w = np.zeros(n_features, dtype=DTYPE)
        self.Ve = np.random.normal(scale=self.init_scale, size=(n_uniques, self.k)).astype(DTYPE)
        self.Vc = np.random.normal(scale=self.init_scale, size=(n_uniques, self.k)).astype(DTYPE)
        self.Vf = np.random.normal(scale=self.init_scale, size=(n_features, self.k)).astype(DTYPE)

    def _regularization_loss(self) -> DTYPE:
        return DTYPE(0.5) * np.sum([
            self.l2_w * np.sum(np.square(self.w)),
            self.l2_V * np.sum(np.square(self.Ve)),
            self.l2_V * np.sum(np.square(self.Vc)),
            self.l2_V * np.sum(np.square(self.Vf))])

    def _update_eta(self, t) -> None:
        """Update learning rate by Robbins-Monro method.
        """
        self.eta *= DTYPE((t + 1) / (t + 2))

    @abstractmethod
    def fit(self) -> None:
        """model fitting
        """

    @abstractmethod
    def predict(self) -> None:
        """make prediction
        """
