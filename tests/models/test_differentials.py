# -*- coding: utf-8 -*-
"""Testing differential functions.

Checking result, type, dtype, and shape.
"""
import numpy as np
from cdfm.config import DTYPE
from cdfm.models import differentials as Diff


class TestDifferentials():
    """Testing equations module in models.
    """

    def setup_method(self, method) -> None:
        """Setup testing context.
        """
        self.Ve = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=DTYPE)
        self.Vc = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=DTYPE)
        self.Vf = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10., 11., 12.]
        ], dtype=DTYPE)

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_Iec_ve(self):
        cinds = [1, 2]
        res = Diff.Iec_ve(cinds, self.Vc)
        expected = np.array([11.0, 13.0, 15.0], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE
        assert res.shape == (3, )

    def test_Iec_vc(self):
        """eind: 0
        """
        eind = 0
        res = Diff.Iec_vc(eind, self.Ve)
        expected = np.array([1.0, 2.0, 3.0], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE
        assert res.shape == (3, )

    def test_p_Iec_u(self):
        eind = 0
        cinds = [1, 2]
        d = np.array([[1, 4], [2, 3]], dtype=DTYPE)
        ps = np.array([0.3, 0.7], dtype=DTYPE)
        res = Diff.p_Iec_u(eind, cinds, self.Ve, self.Vc, d, ps)
        expected = np.array([-27.72, -58.38], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE

    def test_p_Iec_ve(self):
        """eind: 0, cinds: 1, 2
        """
        cinds = [1, 2]
        ps = np.array([0.3, 0.7], dtype=DTYPE)
        res = Diff.p_Iec_ve(cinds, self.Vc, ps)
        expected = np.array([6.1, 7.1, 8.1], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE

    def test_p_Iec_vc(self):
        """eind: 0, (cind: 1), p: 0.3 (between 0 and 1)
        """
        eind = 0
        p = DTYPE(0.3)
        res = Diff.p_Iec_vc(eind, self.Ve, p)
        expected = np.array([0.3, 0.6, 0.9])
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE

    def test_Ief_ve(self):
        x = np.ones(4, dtype=DTYPE)
        res = Diff.Ief_ve(x, self.Vf)
        expected = np.array([22, 26, 30], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE

    def test_Ief_vf(self):
        eind = 0
        x = np.ones(4, dtype=DTYPE)
        res = Diff.Ief_vf(eind, x, self.Vf)
        expected = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ], dtype=DTYPE)
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE

    def test_Iff_vf(self):
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=DTYPE)
        res = Diff.Iff_vf(x, self.Vf)
        expected = np.array([
            [-27.0, -30.0, -33.0],
            [36.0, 36.0, 36.0],
            [-141.0, -156., -171.0],
            [-56.0, -64.0, -72.0],
        ], dtype=DTYPE)  # culculated by hand
        assert np.allclose(res, expected)
        assert isinstance(res, np.ndarray)
        assert res.dtype == DTYPE
