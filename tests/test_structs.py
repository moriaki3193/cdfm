# -*- coding: utf-8 -*-
"""Testing structures.
"""
from os.path import abspath, dirname, join
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
        self.doc1 = Document('a', np.ndarray([1, 1, 1], dtype=DTYPE), DTYPE(1.0))
        self.doc2 = Document('b', np.ndarray([2, 2, 2], dtype=DTYPE), DTYPE(2.0))
        self.doc3 = Document('c', np.ndarray([3, 3, 3], dtype=DTYPE), DTYPE(3.0))
        self.doc4 = Document('a', np.ndarray([4, 4, 4], dtype=DTYPE), DTYPE(4.0))
        self.doc5 = Document('b', np.ndarray([5, 5, 5], dtype=DTYPE), DTYPE(5.0))
        self.doc6 = Document('c', np.ndarray([6, 6, 6], dtype=DTYPE), DTYPE(6.0))
        # Queries
        self.query1 = Query('x', [self.doc1, self.doc2, self.doc3])
        self.query2 = Query('y', [self.doc4, self.doc5, self.doc6])
        # Data
        self.data = Data([self.query1, self.query2])

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_data_from_file(self) -> None:
        # pylint: disable=missing-docstring
        fname = 'sample_train_features.txt'
        p = join(dirname(abspath(__file__)), 'resources', fname)
        data = Data.from_file(p, 4)
        expected = Data([
            Query('1', [
                Document('x', np.array([0.1, -.2, 0.3, 0.0], dtype=DTYPE), DTYPE(0.5)),
                Document('y', np.array([-.1, 0.2, 0.0, 0.4], dtype=DTYPE), DTYPE(0.0)),
                Document('z', np.array([0.0, -.2, 0.3, -.4], dtype=DTYPE), DTYPE(-.5)),
            ]),
            Query('2', [
                Document('y', np.array([0.1, -.2, 0.3, 0.0], dtype=DTYPE), DTYPE(0.5)),
                Document('z', np.array([-.1, 0.2, 0.0, 0.4], dtype=DTYPE), DTYPE(0.0)),
                Document('w', np.array([0.0, -.2, 0.3, -.4], dtype=DTYPE), DTYPE(-.5)),
            ]),
        ])
        assert data == expected

    def test_data_labels(self) -> None:
        # pylint: disable=missing-docstring
        labels = np.array([self.doc1.label, self.doc2.label, self.doc3.label])
        assert np.array_equal(self.query1.labels, labels.astype(DTYPE))

    def test_query_extract_others(self) -> None:
        # pylint: disable=missing-docstring
        assert self.query1.extract_others(self.doc1.id) == [self.doc2.id, self.doc3.id]
