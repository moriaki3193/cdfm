# -*- coding: utf-8 -*-
"""Testing utility functions.
"""
import numpy as np
from cdfm.config import DTYPE
from cdfm.data import CDFMRow
from cdfm.utils import make_map, extract_unique_ids


class TestUtils():
    """Testing utility functions in utils.
    """

    def setup_method(self, method) -> None:
        """Setup testing context.
        """
        self.dataset = [
            CDFMRow(0.5, 'a', 'x', {'y', 'z'}, np.array([1.0, 2.0, 3.0], dtype=DTYPE), None),
            CDFMRow(-.5, 'a', 'y', {'x', 'z'}, np.array([1.0, 2.0, 3.0], dtype=DTYPE), None),
            CDFMRow(0.0, 'a', 'z', {'x', 'y'}, np.array([1.0, 2.0, 3.0], dtype=DTYPE), None),
        ]

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_extract_unique_ids(self) -> None:
        unique_ids = extract_unique_ids(self.dataset)
        expected = {'x', 'y', 'z'}
        assert unique_ids == expected

    def test_make_map(self) -> None:
        unique_ids = extract_unique_ids(self.dataset)
        mapper = make_map(unique_ids)
        assert set(mapper.keys()) == {'x', 'y', 'z'}
        assert set(mapper.values()) == {0, 1, 2}
