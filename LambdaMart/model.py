# coding: utf-8

import xgboost as xgb
import numpy as np

from hyper import Hyper
from utils import dcg

LIMIT_DELTA = 50


params = {
    'max_depth': Hyper.max_depth,
    'eval_metric': 'ndcg@5',
    'learning_rate': Hyper.learning_rate,
}


class LambdaMart(object):
    def __init__(self, num_pairsample=1, param=params):

        self.model = None
        self.params = param
        self.num_pairsample = num_pairsample

        self.train_groups = None
        self.query_id_to_permutations = {}

        self.ndcg = []

    def _initialize(self, dtrain, train_groups):
        print('initialize')
        self.train_groups = train_groups

        dlabels = dtrain.get_label()
        cur_sample = 0

        for i, group in enumerate(train_groups):
            labels = dlabels[cur_sample:cur_sample + group]
            labels_2d = np.tile(labels, (group, 1))

            permutations = np.zeros(shape=(group, group))
            permutations[labels_2d.transpose() > labels_2d] = 1
            permutations[labels_2d.transpose() < labels_2d] = -1

            self.query_id_to_permutations[i] = permutations
            cur_sample += group

    def _objective(self, predict, dtrain):
        dlabels = dtrain.get_label()
        cur_sample = 0

        grad = np.zeros(shape=(predict.shape[0], ))
        hess = np.ones(shape=(predict.shape[0], ))

        self.ndcg.append(0)
        for i, group in enumerate(self.train_groups):
            labels = dlabels[cur_sample:cur_sample + group]
            current_predict = predict[cur_sample:cur_sample + group]

            if np.unique(labels).shape[0] == 1:
                cur_sample += group
                continue

            max_dcg_t = dcg(np.sort(labels)[::-1], 5)
            dcg_t = dcg(labels[np.argsort(current_predict)[::-1]], 5)
            self.ndcg[-1] += dcg_t / max_dcg_t

            permutations = self.query_id_to_permutations[i]

            sorted_ids = np.argsort(current_predict)[::-1] + 1
            max_dcg = dcg(np.sort(labels)[::-1])

            delta_ndcg = np.tile(1 / np.log(sorted_ids + 1), (group, 1))
            delta_ndcg = delta_ndcg.transpose() - delta_ndcg
            delta_ndcg = (np.power(2, labels.reshape((-1, 1))) - 1) * delta_ndcg
            delta_ndcg = delta_ndcg + delta_ndcg.transpose()
            delta_ndcg = np.abs(delta_ndcg / max_dcg)

            ro = np.tile(predict[cur_sample:cur_sample + group], (group, 1))
            ro = np.abs(Hyper.gamma * (ro.transpose() - ro))
            ro = 1 / (np.exp(ro) + 1)

            random_permutation = np.random.permutation(np.arange(1, group**2 + 1)).reshape((group, group))
            random_permutation = random_permutation * np.abs(permutations)
            random_permutation = np.argsort(random_permutation, axis=1)
            random_permutation = random_permutation[:, -self.num_pairsample:]

            _grad = (-Hyper.gamma * ro * delta_ndcg * permutations)[:, random_permutation]
            _hess = (Hyper.gamma**2 * delta_ndcg * ro * (1 - ro) * np.abs(permutations))[:, random_permutation]

            grad[cur_sample:cur_sample + group] = _grad.sum(axis=1).reshape(-1, )
            hess[cur_sample:cur_sample + group] = _hess.sum(axis=1).reshape(-1, )

            cur_sample += group

        hess[np.isclose(hess, 0)] = 1.
        self.ndcg[-1] /= len(self.train_groups)

        return grad, hess

    def fit(self, dtrain, train_groups):
        self._initialize(dtrain, train_groups)

        self.model = xgb.train(
            self.params, dtrain, verbose_eval=True, num_boost_round=Hyper.n_estimators,
            evals=[(dtrain, 'eval')], obj=self._objective,
        )

    def predict(self, dtest, test_groups=None):
        return self.model.predict(dtest)
