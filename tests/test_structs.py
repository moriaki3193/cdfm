# -*- coding: utf-8 -*-
"""Testing structures.
"""
import numpy as np
from cdfm.config import DTYPE
from cdfm.structs import Document, Query, Data


class TestStructs():
    """Testing structures in structs.
    """

    def setup_method(self, _method) -> None:
        """Setup testing context.
        """
        # Documents
        self.doc1 = Document('a', np.ndarray([1, 1, 1], dtype=DTYPE))
        self.doc2 = Document('b', np.ndarray([2, 2, 2], dtype=DTYPE))
        self.doc3 = Document('c', np.ndarray([3, 3, 3], dtype=DTYPE))
        self.doc4 = Document('a', np.ndarray([4, 4, 4], dtype=DTYPE))
        self.doc5 = Document('b', np.ndarray([5, 5, 5], dtype=DTYPE))
        self.doc6 = Document('c', np.ndarray([6, 6, 6], dtype=DTYPE))
        # Queries
        self.query1 = Query('x', [self.doc1, self.doc2, self.doc3])
        self.query2 = Query('y', [self.doc4, self.doc5, self.doc6])
        # Data
        self.data = Data([self.query1, self.query2])

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_data_lookup(self) -> None:
        # pylint: disable=missing-docstring
        # Extract indices of registered documents.
        assert self.data.lookup(self.doc1.id) == self.data.lookup(self.doc4.id)
        assert self.data.lookup(self.doc2.id) == self.data.lookup(self.doc5.id)
        assert self.data.lookup(self.doc3.id) == self.data.lookup(self.doc6.id)
        # Should return len(`Unique Documents`).
        assert self.data.lookup('unregistered') == 3

    def test_query_extract_others(self) -> None:
        # pylint: disable=missing-docstring
        assert self.query1.extract_others(self.doc1.id) == [self.doc2.id, self.doc3.id]
