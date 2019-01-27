# -*- coding: utf-8 -*-
"""Testing rankers.
"""
from os.path import abspath, dirname, join
from cdfm.config import DTYPE
from cdfm.consts import LABEL
from cdfm.utils import load_cdfmdata, build_cdfmdata
from cdfm.models.rankers import CDFMRanker, CDFMRankerV2


DATA_DIR = join(dirname(dirname(abspath(__file__))), 'resources')
RANK_REL_MAP = {
    1.0: 5.0,  # 1st
    2.0: 4.0,  # 2nd
    3.0: 3.0,  # 3rd
    4.0: 2.0,  # 4th
    5.0: 1.0,  # 5th
    6.0: 0.0, 7.0: 0.0, 8.0: 0.0, 9.0: 0.0,
    10.0: 0.0, 11.0: 0.0, 12.0: 0.0, 13.0: 0.0,
    14.0: 0.0, 15.0: 0.0, 16.0: 0.0,
}


class TestCDFMRanker():
    """Testing CDFMRanker class.
    """

    def setup_method(self, method) -> None:
        """Setup testing context.
        """
        self.model = CDFMRanker(k=2,
                                l2_w=1e-2,
                                l2_V=1e-2,
                                n_iter=1000,
                                init_eta=1e-2,
                                init_scale=1e-2)

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_fit(self) -> None:
        features_path = join(DATA_DIR, 'sample_features.txt')
        features = load_cdfmdata(features_path, 4)
        train = build_cdfmdata(features)
        self.model.fit(train, verbose=False)
        print(self.model.scores)
        assert self.model.w.dtype == DTYPE
        assert self.model.Ve.dtype == DTYPE
        assert self.model.Vc.dtype == DTYPE
        assert self.model.Vf.dtype == DTYPE

    def test_predict(self) -> None:
        pass


class TestCDFMRankerV2(TestCDFMRanker):
    """Testing CDFMRankerV2 class.
    """

    def setup_method(self, method) -> None:
        super().setup_method(method)
