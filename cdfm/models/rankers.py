# -*- coding: utf-8 -*-
"""Implementation of Learning to Rank models.
"""
import random
from typing import List
import numpy as np
from fastprogress import master_bar, progress_bar
from ..config import DTYPE
from .base import CDFMRankerMeta
from . import equations as Eqn
from . import differentials as Diff
from ..data import CDFMRow
from ..types import CDFMDataset
from ..utils import _extract_first


class CDFMRanker(CDFMRankerMeta):
    """Combination-dependent Entity Ranking.
    """

    def _model_equation(self, eind: int, cinds: List[int], x: np.ndarray) -> DTYPE:
        iec = Eqn.Iec(eind, cinds, self.Ve, self.Vc)
        ief = Eqn.Ief(eind, x, self.Ve, self.Vf)
        iff = Eqn.Iff(x, self.Vf)
        return self.b + np.dot(self.w, x) + iec + ief + iff

    def _update_params(self, err: DTYPE, eind: int, cinds: List[int], x: np.ndarray) -> DTYPE:
        step: DTYPE = self.eta * err
        coef_b = DTYPE(1.)
        coef_w: np.ndarray = x \
            - np.multiply(self.l2_w, self.w) \
            - np.multiply(self.l2_w, self.w)
        coef_Ve: np.ndarray = Diff.Iec_ve(cinds, self.Vc) \
            + Diff.Ief_ve(x, self.Vf) \
            - np.multiply(self.l2_V, self.Ve[eind])
        coef_Vc: np.ndarray = Diff.Iec_vc(eind, self.Ve) \
            - np.multiply(self.l2_V, self.Vc[cinds])
        coef_Vf: np.ndarray = Diff.Ief_vf(eind, x, self.Ve) \
            + Diff.Iff_vf(x, self.Vf) \
            - np.multiply(self.l2_V, self.Vf)
        self.b -= np.multiply(step, coef_b)
        self.w -= np.multiply(step, coef_w)
        self.Ve[eind] -= np.multiply(step, coef_Ve)
        self.Vc[cinds] -= np.multiply(step, coef_Vc)
        self.Vf -= np.multiply(step, coef_Vf)

    def fit(self, data: CDFMDataset, verbose: bool = True) -> None:
        """Training the model.

        Parameters:
            data: CDFMDataset instance.
            verbose: Whether display training processes.
        """
        self._init_params(data)
        n_samples = len(data)
        _indices: List[int] = list(range(n_samples))
        # model fitting
        parent_bar = master_bar(range(self.n_iter)) if verbose else range(self.n_iter)
        for t in parent_bar:
            pred_labels: np.ndarray = np.empty(n_samples, dtype=DTYPE)
            obs_labels: np.ndarray = np.empty(n_samples, dtype=DTYPE)
            child_bar = progress_bar(_indices, parent=parent_bar) if verbose else _indices
            for i in child_bar:
                # 1. predict
                row: CDFMRow = data[i]
                eind = self.map[row[self._eid_loc]]
                cinds = [self.map[cid] for cid in row[self._cids_loc]]
                x = row[self._feat_loc]
                pred_labels[i] = self._model_equation(eind, cinds, x)
                obs_labels[i] = row[self._label_loc]
                # 2. update
                pred_err = pred_labels[i] - obs_labels[i]
                self._update_params(pred_err, eind, cinds, x)
                if verbose:
                    parent_bar.child.comment = f'Epoch {t + 1}'
            self.scores[t] = self._evaluate_score(pred_labels, obs_labels)
            self._update_eta(t)
            random.shuffle(_indices)
            if verbose:
                parent_bar.write(f'Epoch {t + 1} (Score: {np.round(self.scores[t], 5)})')

    def predict(self, data: CDFMDataset) -> np.ndarray:
        """Make prediction on a given data.

        Parameters:
            data: CDFMDataset instance.

        Returns:
            pred_labels: an array of predicted labels. shape (#samples, )
        """
        pred_labels: np.ndarray = np.empty(len(data), dtype=DTYPE)
        for i, row in enumerate(data):
            eind = self.map[row[self._eid_loc]]
            cinds = [self.map[cid] for cid in row[self._cids_loc]]
            x = row[self._feat_loc]
            pred_labels[i] = self._model_equation(eind, cinds, x)
        return pred_labels


class CDFMRankerV2(CDFMRankerMeta):
    """Combination-dependent Entity Ranking with Proximities.
    """

    def __init__(self,
                 k: int = 10,
                 l2_u: float = 1e-2,
                 l2_w: float = 1e-2,
                 l2_V: float = 1e-2,
                 n_iter: int = 1000,
                 init_eta: float = 1e-2,
                 init_scale: float = 1e-2) -> None:
        model_config = {'k': k,
                        'l2_w': l2_w,
                        'l2_V': l2_V,
                        'n_iter': n_iter,
                        'init_eta': init_eta,
                        'init_scale': init_scale}
        super().__init__(**model_config)
        # proximity factors and its regularization term.
        self.l2_u = DTYPE(l2_u)
        self.u: np.ndarray

    def _init_params(self, data: CDFMDataset) -> None:
        super()._init_params(data)
        first_row = _extract_first(data)
        _, n_factors = first_row[self._prox_loc].shape
        self.u = np.zeros(n_factors, dtype=DTYPE)

    def _regularization_loss(self) -> DTYPE:
        pass

    def fit(self) -> None:
        """Training the model.

        Parameters:
            data: DataFrame whose columns are (LABEL, QID, EID, FEATURES).
            verbose: Whether display training processes.
        """
        pass

    def predict(self) -> None:
        pass
