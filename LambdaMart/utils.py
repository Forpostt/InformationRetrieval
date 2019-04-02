# coding: utf-8

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_svmlight_file

from hyper import Hyper


def load_data():
    print('load and create train Dmatrix')
    x_train, y_train, q_id_train = load_svmlight_file(Hyper.train_data, query_id=True)
    groups_train = np.bincount(q_id_train)[q_id_train[0]:]

    train_dmatrix = xgb.DMatrix(x_train, y_train)
    train_dmatrix.set_group(groups_train)

    print('load and create test Dmatrix')
    x_test, y_test, q_id_test = load_svmlight_file(Hyper.test_data, query_id=True)
    groups_test = np.bincount(q_id_test)[q_id_test[0]:]

    test_dmatrix = xgb.DMatrix(x_test, y_test)
    test_dmatrix.set_group(groups_test)

    return (train_dmatrix, groups_train, q_id_train), (test_dmatrix, groups_test, q_id_test)


def create_submission(q_id_test, predict):
    print('create submission')

    with open(Hyper.submission, 'w') as fd:
        fd.write('QueryId,DocumentId\n')

        for q_id in np.unique(q_id_test):
            doc_idx = np.where(q_id_test == q_id)[0]
            sorted_args = np.argsort(predict[doc_idx])[::-1] + doc_idx.min()

            for doc_id in sorted_args:
                fd.write('{},{}\n'.format(q_id, doc_id + 1))


def dcg(labels, t=None):
    if t is None:
        t = len(labels)
    return ((np.power(2, labels) - 1) / np.log(np.arange(1, len(labels) + 1) + 1))[:t].sum()
