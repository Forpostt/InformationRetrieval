# coding: utf-8


class Hyper(object):
    base_path = '/home/pmankevich/git/InformationRetrieval/LambdaMart'

    train_data = '{}/data/train.txt'.format(base_path)
    test_data = '{}/data/test.txt'.format(base_path)

    n_estimators = 10
    max_depth = 8
    learning_rate = 0.3
    gamma = 1.0

    submission = '{}/data/sub.csv'.format(base_path)
