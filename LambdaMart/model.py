# coding: utf-8

import xgboost as xgb
import numpy as np
import tqdm

from hyper import Hyper
from utils import dcg


params = {
    'eta': 0.2,
    'max_depth': Hyper.max_depth,
    'eval_metric': 'ndcg@5',
}


class LambdaMart(object):
    def __init__(self, param=params):

        self.model = None
        self.params = param

        self.train_groups = None
        self.query_id_to_permutations = {}

    def _initialize(self, dtrain, train_groups):
        print('initialize')

        self.train_groups = train_groups

        dlabels = dtrain.get_label()
        cur_sample = 0

        for i, group in tqdm.tqdm(enumerate(train_groups)):
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

        grad = np.ones(shape=(predict.shape[0], ))
        hess = np.ones(shape=(predict.shape[0], ))

        for i, group in enumerate(self.train_groups):
            labels = dlabels[cur_sample:cur_sample + group]

            if np.unique(labels).shape[0] == 1:
                cur_sample += group
                continue

            current_predict = predict[cur_sample:cur_sample + group]
            permutations = self.query_id_to_permutations[i]

            max_dcg = dcg(np.sort(labels)[::-1])
            sorted_ids = np.argsort(current_predict)[::-1] + 1

            delta_ndcg = np.tile(1 / np.log(sorted_ids + 1), (group, 1))
            delta_ndcg = delta_ndcg.transpose() - delta_ndcg
            delta_ndcg = (np.power(2, labels.reshape((-1, 1))) - 1) * delta_ndcg
            delta_ndcg = delta_ndcg + delta_ndcg.transpose()
            delta_ndcg = np.abs(delta_ndcg / max_dcg)

            ro = np.tile(predict[cur_sample:cur_sample + group], (group, 1))
            ro = Hyper.gamma * (ro.transpose() - ro)
            ro = 1 / (np.exp(ro) + 1)

            grad[cur_sample:cur_sample + group] = (-Hyper.gamma * ro * delta_ndcg * permutations).sum(axis=1).reshape(-1, )
            hess[cur_sample:cur_sample + group] = (Hyper.gamma**2 * delta_ndcg * ro * (1 -ro) * np.abs(permutations)).sum(axis=1).reshape(-1, )

            cur_sample += group

        return grad.reshape((-1, 1)), hess.reshape((-1, 1))

    def fit(self, dtrain, train_groups):
        self._initialize(dtrain, train_groups)

        self.model = xgb.train(
            self.params, dtrain, verbose_eval=True, num_boost_round=Hyper.n_estimators,
            evals=[(dtrain, 'eval')], obj=self._objective,
        )

    def predict(self, dtest, test_groups=None):
        return self.model.predict(dtest)


if __name__ == '__main__':
    model = LambdaMart()
