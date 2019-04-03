# coding: utf-8

import time
import xgboost as xgb
import numpy as np

from hyper import Hyper
from utils import dcg, load_data, create_submission


def objective(train_dmatrix, train_groups):
    print('initialize')

    dlabels = train_dmatrix.get_label()

    query_id_to_docs_comparison = {}                                   # Dict with all possible doc pairs for each query
    cur_sample_id = 0

    for i, group_len in enumerate(train_groups):
        labels = dlabels[cur_sample_id:cur_sample_id + group_len]
        labels_2d = np.tile(labels, (group_len, 1))

        docs_comparison = np.zeros(shape=(group_len, group_len))
        docs_comparison[labels_2d.transpose() > labels_2d] = 1         # Doc i is better than j -> permutations[i, j] = 1
        docs_comparison[labels_2d.transpose() < labels_2d] = -1        # Doc i is worse than j -> permutations[i, j] = -1

        query_id_to_docs_comparison[i] = docs_comparison
        cur_sample_id += group_len

    def _objective(predict, dtrain):
        cur_sample = 0

        grad = np.zeros(shape=(predict.shape[0], ))
        hess = np.ones(shape=(predict.shape[0], ))

        for i, group in enumerate(train_groups):
            group_labels = dlabels[cur_sample:cur_sample + group]
            current_predict = predict[cur_sample:cur_sample + group]
            docs_comparison = query_id_to_docs_comparison[i]

            if np.unique(group_labels).shape[0] == 1:
                cur_sample += group
                continue

            sorted_ids = np.argsort(current_predict)[::-1] + 1
            max_dcg = dcg(np.sort(group_labels)[::-1])

            delta_ndcg = (1 / np.log(sorted_ids + 1)).reshape((-1, 1)) - (1 / np.log(sorted_ids + 1))
            delta_ndcg = (2**group_labels.reshape((-1, 1)) - 2**group_labels) * delta_ndcg
            delta_ndcg = np.abs(delta_ndcg / max_dcg)

            ro = current_predict.reshape((-1, 1)) - current_predict
            ro = np.abs(Hyper.gamma * (ro.transpose() - ro))
            ro = 1 / (np.exp(ro) + 1)

            # Get only one random pair for each document, like in xgboost
            random_doc_pairs = np.random.permutation(np.arange(1, group**2 + 1)).reshape((group, group))
            random_doc_pairs = random_doc_pairs * np.abs(docs_comparison)
            random_doc_pairs = np.argsort(random_doc_pairs, axis=1)
            random_doc_pairs = random_doc_pairs[:, -1:]

            cropped_docs_comparison = np.zeros(shape=(group, group))
            cropped_docs_comparison[range(group), random_doc_pairs] = docs_comparison[range(group), random_doc_pairs]
            cropped_docs_comparison[random_doc_pairs, range(group)] = docs_comparison[random_doc_pairs, range(group)]

            _grad = -Hyper.gamma * ro * delta_ndcg * cropped_docs_comparison
            _hess = Hyper.gamma**2 * delta_ndcg * ro * (1 - ro) * np.abs(cropped_docs_comparison)

            grad[cur_sample:cur_sample + group] = _grad.sum(axis=1).reshape(-1, )
            hess[cur_sample:cur_sample + group] = _hess.sum(axis=1).reshape(-1, )

            cur_sample += group

        hess[np.isclose(hess, 0)] = 1e-6
        return grad, hess

    return _objective


def xgboost_():
    _start = time.time()
    train, test = load_data()

    params = {
        'eta': Hyper.learning_rate,
        'max_depth': Hyper.max_depth,
        'eval_metric': 'ndcg@5',
    }
    obj = objective(train[0], train[1])

    model = xgb.train(
        params, train[0], verbose_eval=True, num_boost_round=Hyper.n_estimators,
        evals=[(train[0], 'eval')], obj=obj,
    )

    print('total time: {}'.format(time.time() - _start))
    create_submission(test[2], model.predict(test[0]))


if __name__ == '__main__':
    xgboost_()
