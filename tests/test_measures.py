# -*- coding: utf-8 -*-
"""Testing evaluation measures.
"""
import numpy as np
from cdfm.config import DTYPE
from cdfm.measures import compute_RMSE


class TestMeasures():
    """Testing evaluation functions in measures.
    """

    def setup_method(self, method) -> None:
        """Setup testing context.
        """
        self.pred_labels = np.array([2., 4., 10.], dtype=DTYPE)
        self.obs_labels = np.array([3., 5., 8.], dtype=DTYPE)

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_compute_RMSE(self) -> None:
        res = compute_RMSE(self.pred_labels, self.obs_labels)
        expected = np.sqrt(2.0)
        assert np.isclose(res, expected)
        assert isinstance(res, DTYPE)
