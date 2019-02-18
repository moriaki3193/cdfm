# -*- coding: utf-8 -*-
"""Implementation of Learning to Rank models.
"""
import random
import itertools
import multiprocessing as mp
from copy import deepcopy
from functools import partial
from typing import Callable, List, Tuple
import numpy as np
from fastprogress import master_bar, progress_bar
from ..config import DTYPE
from .base import CDFMRankerMeta
from . import equations as Eqn
from . import differentials as Diff
from ..data import CDFMRow
from ..structs import Data, Query, Document
from ..types import CDFMDataset, DocumentID, DocIdxMap
from ..utils import _extract_first


CDERankerParams = Tuple[DTYPE,       # b
                        np.ndarray,  # w
                        np.ndarray,  # Ve
                        np.ndarray,  # Vc
                        np.ndarray]  # Vf


class CDERanker:
    """Combination-dependent Entity Ranker.

    Attributes:
        k: #dimensions of latent vectors.
        l2_w: L2 regularization on pointwise weights.
        l2_Ve: L2 regularization on entity latent vectors.
        l2_Vc: L2 regularization on competitor latent vectors.
        l2_Vf: L2 regularization on feature latent vectors.
        max_iter: #max-iterations when fitting.
        eta: initial learning rate.
        init_scale: v ~ Normal(0., scale=init_scale).
        metric: evaluation measure.

    Methods:
        fit: model fitting using minibatch Gradient Descent algorithm.
        predict: make prediction on a given dataset.
    """

    def __init__(self,
                 k: int = 8,
                 l2_w: float = 1e-2,
                 l2_Ve: float = 1e-1,
                 l2_Vc: float = 1e-1,
                 l2_Vf: float = 1e-2,
                 max_iter: int = 100,
                 init_eta: float = 1e-2,
                 init_scale: float = 1e-2,
                 metric: str = 'NDCG@5') -> None:
        self.k = k
        self.l2_w = DTYPE(l2_w)
        self.l2_Ve = DTYPE(l2_Ve)
        self.l2_Vc = DTYPE(l2_Vc)
        self.l2_Vf = DTYPE(l2_Vf)
        self.max_iter = max_iter
        self.eta = DTYPE(init_eta)
        self.init_scale = DTYPE(init_scale)
        self.metric = metric
        # Variables set dynamically
        self.map: DocIdxMap
        self.b: DTYPE
        self.w: np.ndarray
        self.Ve: np.ndarray
        self.Vc: np.ndarray
        self.Vf: np.ndarray

    def _init_map(self, data: Data) -> None:
        lol = [[d.id for d in q.docs] for q in data.queries]
        uniques = set(itertools.chain.from_iterable(lol))
        self.map = {doc_id: idx for idx, doc_id in enumerate(uniques)}

    def _init_params(self, data: Data) -> None:
        first_query = _extract_first(data.queries)
        first_doc = _extract_first(first_query.docs)
        p = len(self.map) + 1
        (q, ) = first_doc.vec.shape
        self.b = DTYPE(0.)
        self.w = np.zeros(q, dtype=DTYPE)
        self.Ve = np.random.normal(scale=self.init_scale, size=(p, self.k)).astype(DTYPE)
        self.Vc = np.random.normal(scale=self.init_scale, size=(p, self.k)).astype(DTYPE)
        self.Vf = np.random.normal(scale=self.init_scale, size=(q, self.k)).astype(DTYPE)

    @profile
    def _update_params(self, query: Query) -> None:
        # pylint: disable=too-many-locals
        Ve = deepcopy(self.Ve)
        Vc = deepcopy(self.Vc)
        Vf = deepcopy(self.Vf)
        pred_err: np.ndarray = self._minibatch_predict(query) - query.labels
        docidx = [self.map.get(d.id, len(self.map)) for d in query.docs]
        step = self.eta / DTYPE(len(query.docs))
        for i, (err, doc) in enumerate(zip(pred_err, query.docs)):
            # compute nabra
            eidx = docidx[i]
            cidx = [idx for h, idx in enumerate(docidx) if h != i]
            coef_Ve = Diff.Iec_ve(cidx, Vc) + Diff.Ief_ve(doc.vec, Vf)
            coef_Vc = Diff.Iec_vc(eidx, Ve)
            coef_Vf = Diff.Ief_vf(eidx, doc.vec, Ve) + Diff.Iff_vf(doc.vec, Vf)
            nabra_b = err * DTYPE(1.0)
            nabra_w = err * doc.vec + self.l2_w * doc.vec
            nabra_Ve = err * coef_Ve + self.l2_Ve * Ve[eidx]
            nabra_Vc = err * coef_Vc + self.l2_Vc * Vc[cidx]
            nabra_Vf = err * coef_Vf + self.l2_Vf * Vf
            # update params
            self.b -= np.multiply(step, nabra_b)
            self.w -= np.multiply(step, nabra_w)
            self.Ve[eidx] -= np.multiply(step, nabra_Ve)
            self.Vc[cidx] -= np.multiply(step, nabra_Vc)
            self.Vf -= np.multiply(step, nabra_Vf)

    @profile
    def _minibatch_predict(self, query: Query) -> np.ndarray:
        pred_labels: List[DTYPE] = []
        docidx = [self.map.get(d.id, len(self.map)) for d in query.docs]
        for i, doc in enumerate(query.docs):
            eidx = docidx[i]
            cidx = [idx for h, idx in enumerate(docidx) if h != i]
            iec = Eqn.Iec(eidx, cidx, self.Ve, self.Vc)
            ief = Eqn.Ief(eidx, doc.vec, self.Ve, self.Vf)
            iff = Eqn.Iff(doc.vec, self.Vf)
            pred_labels.append(self.b + np.dot(self.w, doc.vec) + iec + ief + iff)
        return np.array(pred_labels, dtype=DTYPE)

    def fit(self, data: Data, verbose: bool = True) -> None:
        """Training the model.

        Parameters:
            data: Data instance.
            verbose: Whether display training processes.
        """
        self._init_map(data)
        self._init_params(data)
        _indices: List[int] = list(range(len(data.queries)))
        mbar = master_bar(range(self.max_iter) if verbose else range(self.max_iter))
        # t: epoch indexer, n: query indexer
        for t in mbar:
            pbar = progress_bar(_indices, parent=mbar) if verbose else _indices
            for n in pbar:
                self._update_params(data.queries[n])
                if verbose:
                    mbar.child.comment = f'Epoch {t + 1}'
            random.shuffle(_indices)
            score = DTYPE(0.0)
            if verbose:
                mbar.write(f'Epoch {t + 1} Avg. {self.metric}: {np.round(score, 5)}')

    def predict(self) -> None:
        pass


class CDFMRanker(CDFMRankerMeta):
    """Combination-dependent Entity Ranking.
    """

    def _model_equation(self, eind: int, cinds: List[int], x: np.ndarray) -> DTYPE:
        iec = Eqn.Iec(eind, cinds, self.Ve, self.Vc)
        ief = Eqn.Ief(eind, x, self.Ve, self.Vf)
        iff = Eqn.Iff(x, self.Vf)
        return self.b + np.dot(self.w, x) + iec + ief + iff

    def _update_params(self, err: DTYPE, eind: int, cinds: List[int], x: np.ndarray) -> None:
        step: DTYPE = self.eta * err
        coef_b = DTYPE(1.)
        coef_w: np.ndarray = x \
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
                parent_bar.write(f'Epoch {t + 1} [Score: {np.round(self.scores[t], 5)}]')

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
        n_factors = len(first_row[self._prox_loc][0])
        self.u = np.zeros(n_factors, dtype=DTYPE)

    def _update_params(self,
                       err: DTYPE,
                       eind: int,
                       cinds: List[int],
                       x: np.ndarray,
                       d: List[np.ndarray],
                       ps: np.ndarray) -> None:
        """Parameter updation.

        Variables:
            coef_*: the result of partial differentiation over squared loss.
            nabra_*: the result of partial differentiation over loss function.
        """
        # convert list of ndarray to 2-d ndarray
        darr: np.ndarray = np.array(d, dtype=DTYPE)
        # w
        nabra_w = np.multiply(err, x) + np.multiply(self.l2_w, self.w)
        # u
        coef_u = Diff.p_Iec_u(eind, cinds, self.Ve, self.Vc, darr, ps)
        nabra_u = np.multiply(err, coef_u) + np.multiply(self.l2_u, self.u)
        # Ve
        coef_Ve = Diff.p_Iec_ve(cinds, self.Vc, ps) + Diff.Ief_ve(x, self.Vf)
        nabra_Ve = np.multiply(err, coef_Ve) + np.multiply(self.l2_V, self.Ve[eind])
        # Vf
        coef_Vf = Diff.Ief_vf(eind, x, self.Ve) + Diff.Iff_vf(x, self.Vf)
        nabra_Vf = np.multiply(err, coef_Vf) + np.multiply(self.l2_V, self.Vf)
        # Vc & update
        for proba, cind in zip(ps, cinds):
            coef_Vc = Diff.p_Iec_vc(eind, self.Ve, proba)
            nabra_Vc = np.multiply(err, coef_Vc) + np.multiply(self.l2_V, self.Vc[cind])
            self.Vc[cind] -= np.multiply(self.eta, nabra_Vc)
        # update other params
        self.b -= self.eta * err * DTYPE(1.)
        self.w -= np.multiply(self.eta, nabra_w)
        self.u -= np.multiply(self.eta, nabra_u)
        self.Ve[eind] -= np.multiply(self.eta, nabra_Ve)
        self.Vf -= np.multiply(self.eta, nabra_Vf)

    def __pred_proba(self, p: np.ndarray) -> DTYPE:
        return DTYPE(1. / (1. + np.exp(-np.dot(self.u, p))))

    def _pred_probas(self, d: List[np.ndarray]) -> np.ndarray:
        return np.array([self.__pred_proba(p) for p in d], dtype=DTYPE)

    def _model_equation(self,
                        eind: int,
                        cinds: List[int],
                        x: np.ndarray,
                        ps: np.ndarray) -> DTYPE:
        iec = Eqn.p_Iec(eind, cinds, self.Ve, self.Vc, ps)
        ief = Eqn.Ief(eind, x, self.Ve, self.Vf)
        iff = Eqn.Iff(x, self.Vf)
        return self.b + np.dot(self.w, x) + iec + ief + iff

    def fit(self, data: CDFMDataset, verbose: bool = True) -> None:
        """Training the model.

        Parameters:
            data: DataFrame whose columns are (LABEL, QID, EID, FEATURES).
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
                d = row[self._prox_loc]
                ps = self._pred_probas(d)
                pred_labels[i] = self._model_equation(eind, cinds, x, ps)
                obs_labels[i] = row[self._label_loc]
                # 2. update
                pred_err = pred_labels[i] - obs_labels[i]
                self._update_params(pred_err, eind, cinds, x, d, ps)
                if verbose:
                    parent_bar.child.comment = f'Epoch {t + 1}'
            self.scores[t] = self._evaluate_score(pred_labels, obs_labels)
            self._update_eta(t)
            random.shuffle(_indices)
            if verbose:
                parent_bar.write(f'Epoch {t + 1} [Score: {np.round(self.scores[t], 5)}]')

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
            d = row[self._prox_loc]
            ps = self._pred_probas(d)
            pred_labels[i] = self._model_equation(eind, cinds, x, ps)
        return pred_labels
