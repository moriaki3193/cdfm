# -*- coding: utf-8 -*-
"""Testing equation functions.
"""
import numpy as np
from cdfm.config import DTYPE
from cdfm.models import equations as Eqn


class TestEquations(object):
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
        pass

    def test_Iec(self) -> None:
        """eind: 0, cinds: 1, 2
        """
        eind = 0
        cinds = [1, 2]
        res = Eqn.Iec(eind, cinds, self.Ve, self.Vc)
        expected = 82.0
        assert res == expected
        assert isinstance(res, DTYPE)

    def test_Ief(self) -> None:
        """eind: 0
        """
        eind = 0
        x = np.ones(4, dtype=DTYPE)
        res = Eqn.Ief(eind, x, self.Ve, self.Vf)
        expected = 164.0
        assert res == expected
        assert isinstance(res, DTYPE)

    def test_Iff(self) -> None:
        """with a bit complicated feature.
        """
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=DTYPE)
        res = Eqn.Iff(x, self.Vf)
        expected = -2774.0
        assert res == expected
        assert isinstance(res, DTYPE)
