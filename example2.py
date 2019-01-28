# -*- coding: utf-8 -*-
"""Combination-dependent Learnig to Rank sample script.
"""
import pickle
from os import path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from cdfm.config import DTYPE
from cdfm.consts import LABEL, QID, EID, FEATURES
from cdfm.models.rankers import CDFMRankerV2
from cdfm.measures import compute_RMSE


LABEL_REL_MAP = {
    1.0: 5.0,  # 1st
    2.0: 4.0,  # 2nd
    3.0: 3.0,  # 3rd
    4.0: 2.0,  # 4th
    5.0: 1.0,  # 5th
    6.0: 0.0, 7.0: 0.0, 8.0: 0.0, 9.0: 0.0,
    10.0: 0.0, 11.0: 0.0, 12.0: 0.0, 13.0: 0.0,
    14.0: 0.0, 15.0: 0.0, 16.0: 0.0,
}

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), 'tests', 'resources', 'horseracing')
DUMP_DIR = path.join(path.dirname(path.abspath(__file__)), 'dumps')
N_FEATURES = 16
N_FACTORS = 2

parser = ArgumentParser(description='cdfm sample script.')
parser.add_argument('--k', type=int, default=2, help='#dimensions of latent vectors.')
parser.add_argument('--n-iter', type=int, default=1000, help='#iterations of training.')


if __name__ == '__main__':
    cmdargs = vars(parser.parse_args())

    # loading
    print('loading datasets...')
    with open('./dumps/dataset/train.pkl', mode='rb') as fp:
        train = pickle.load(fp)
    with open('./dumps/dataset/test.pkl', mode='rb') as fp:
        test = pickle.load(fp)

    # fitting
    print('model fitting...')
    k = cmdargs['k']
    n_iter = cmdargs['n_iter']
    model = CDFMRankerV2(k=k, n_iter=n_iter, init_eta=1e-3, init_scale=1e-2)
    model.fit(train, verbose=True)

    # prediction
    print('\nprediction examples...\n')
    group_ids = np.array([row[1] for row in test], dtype=DTYPE).astype(str)
    entity_ids = np.array([row[2] for row in test], dtype=DTYPE).astype(str)
    obs_labels = np.array([row[0] for row in test], dtype=DTYPE)
    pred_labels = model.predict(test)

    # writing out predictions
    outp = path.join(DUMP_DIR, 'predictions', f'v2-pred-k{k}-iter{n_iter}.csv')
    pd.DataFrame({'obs_label': obs_labels,
                  'pred_label': pred_labels,
                  'qid': group_ids,
                  'eid': entity_ids}).to_csv(outp, index=False)
    print(f'\npredictions : {outp}')

    # dumping
    dump_path = path.join(DUMP_DIR, 'models', f'v2-model-k{k}-iter{n_iter}.pkl')
    with open(dump_path, mode='wb') as fp:
        pickle.dump(model, fp)
    print(f'\nfitted model : {dump_path}')

    # show result
    print(f'\nTrain Score: {model.scores[-1]}')
    print(f'Test Score: {compute_RMSE(pred_labels, obs_labels)}')
