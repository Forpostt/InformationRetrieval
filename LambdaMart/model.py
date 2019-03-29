# coding: utf-8

import xgboost as xgb

from hyper import Hyper
from utils import load_data, create_submission


params = {
    'gamma': 1.0,
    'learning_rate': Hyper.learning_rate,
    'max_depth': Hyper.max_depth,
    'min_child_weight': 0.1,
    'objective': 'rank:pairwise',
    'eval_metric': ['ndcg@5', 'map@5'],
}


def model():
    train_dmatrix, test_dmatrix, q_id_test = load_data()

    print('fit model')
    xgb_model = xgb.train(
        params, train_dmatrix, verbose_eval=True, num_boost_round=Hyper.n_estimators,
        evals=[(train_dmatrix, 'eval')],
    )
    predict = xgb_model.predict(test_dmatrix)

    create_submission(q_id_test, predict)


if __name__ == '__main__':
    model()
