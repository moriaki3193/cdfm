# -*- coding: utf-8 -*-
"""Evalutate model output.
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from irmet import NDCG


parser = ArgumentParser(description='evaluate cdfm model output.')
parser.add_argument('--measure', type=str, help='a evaluation measure.')
parser.add_argument('--topk', type=int, default=5, help='k for NDCG@k, ERR@k, and so on.')
parser.add_argument('--path', type=str, help='output file path.')


if __name__ == '__main__':
    cmdargs = vars(parser.parse_args())
    path = cmdargs['path']
    topk = cmdargs['topk']

    # load the file
    df = pd.read_csv(path)

    # algorithm
    # 1. group dataframe by 'qid'.
    # 2. sort records by 'pred_values' in a group.
    # 3. extract relevance score lists.
    # 4. compute the evaluation measures.
    # 5. output the average of the measures.
    scores = []
    for qid, group in df.groupby('qid'):
        sorted_records = group.sort_values('pred_label', ascending=False)
        rels: np.ndarray = sorted_records['obs_label'].values
        # TODO control sequence by 'measure' cdmarg.
        score = NDCG(rels, topk=topk)
        scores.append(score)

    print(f'Score: {np.mean(scores)}, (±{np.std(scores, ddof=1)})')
