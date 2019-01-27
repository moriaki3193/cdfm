# -*- coding: utf-8 -*-
"""Testing rankers.
"""
from os.path import abspath, dirname, join
from cdfm.config import DTYPE
from cdfm.utils import load_cdfmdata, build_cdfmdata
from cdfm.models.rankers import CDFMRanker, CDFMRankerV2


DATA_DIR = join(dirname(dirname(abspath(__file__))), 'resources')


class TestCDFMRanker():
    """Testing CDFMRanker class.
    """

    def setup_method(self, method) -> None:
        """Setup testing context.
        """
        h_params = {'k': 2, 'l2_w': 1e-2, 'l2_V': 1e-2,
                    'n_iter': 100, 'init_eta': 1e-1, 'init_scale': 1e-2}
        self.model = CDFMRanker(**h_params)

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_fit_pred(self) -> None:
        features_path = join(DATA_DIR, 'sample_train_features.txt')
        features = load_cdfmdata(features_path, 4)
        train = build_cdfmdata(features)
        self.model.fit(train, verbose=False)
        # test fit method
        assert self.model.w.dtype == DTYPE
        assert self.model.Ve.dtype == DTYPE
        assert self.model.Vc.dtype == DTYPE
        assert self.model.Vf.dtype == DTYPE
        # test predict method
        features_path = join(DATA_DIR, 'sample_test_features.txt')
        features = load_cdfmdata(features_path, 4)
        test = build_cdfmdata(features)
        pred_labels = self.model.predict(test)
        assert pred_labels.dtype == DTYPE


class TestCDFMRankerV2(TestCDFMRanker):
    """Testing CDFMRankerV2 class.
    """

    def setup_method(self, method) -> None:
        super().setup_method(method)
        h_params = {'k': 2, 'l2_w': 1e-2, 'l2_V': 1e-2,
                    'n_iter': 100, 'init_eta': 1e-1, 'init_scale': 1e-2}
        self.model = CDFMRankerV2(**h_params)

    def teardown_method(self, method) -> None:
        """Clean up testing context.
        """

    def test_fit_pred(self) -> None:
        pass
