# coding: utf-8

import os
import tqdm
import time

from hyper import Hyper
from models import BM25, YandexModel, Lsa
from parser import TITLE_SEP

EMPTY_DOC = ['<EMPTY_DOC_TOKEN>']
EMPTY_TITLE = ['<EMPTY_TITLE_TOKEN>']


def load_corpus():
    titles = []
    docs = []

    processed_files = set(os.listdir(Hyper.processed_content_folder))
    with open(Hyper.sorted_docs) as fd:
        for line in tqdm.tqdm(fd):
            _, file = line.strip().split('\t')

            if file not in processed_files:
                titles.append(EMPTY_TITLE)
                docs.append(EMPTY_DOC)
                continue

            with open(os.path.join(Hyper.processed_content_folder, file)) as doc_fd:
                title, doc = doc_fd.read().strip().split(TITLE_SEP)

                titles.append(title.split(' '))
                docs.append(doc.split(' '))

    return titles, docs


def bm25():
    print('train bm25')
    _, docs = load_corpus()
    bm_25 = BM25(docs)
    bm_25.process()


def yandex():
    print('train yandex')
    yandex_model = YandexModel()
    yandex_model.process()


def lsa():
    _start = time.time()
    print('train lsa')
    lsa_model = Lsa.load()
    lsa_model.load_corpus(split=False)
    # lsa_model.fit()
    lsa_model.process()
    print('fit time: {}'.format(time.time() - _start))


if __name__ == '__main__':
    yandex()
