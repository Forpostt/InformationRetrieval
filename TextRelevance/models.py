# coding: utf-8

import collections
import math
import os
import pickle
import tqdm

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from gensim.summarization.bm25 import BM25 as gensim_bm25

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hyper import Hyper
from parser import TITLE_SEP

DocScore = collections.namedtuple('DocScore', ('id', 'score'))
QueryResult = collections.namedtuple('QueryResult', ('id', 'results'))
ModelResult = collections.namedtuple('ModelResult', ('queries', ))

TOTAL_WORDS_IN_DOC = 'TOTAL_WORDS_IN_DOC'
TOTAL_WORDS_IN_CORPUS = 'TOTAL_WORDS_IN_CORPUS'
EMPTY_LSA_DOC = '<EMPTY_DOC_TOKEN>'

EMPTY_DOC = ['<EMPTY_DOC_TOKEN>']
EMPTY_TITLE = ['<EMPTY_TITLE_TOKEN>']


class BaseModel(object):
    results_folder = Hyper.base_results

    def __init__(self):
        self.titles = []
        self.corpus = []

    def load_corpus(self, split=True):
        processed_files = set(os.listdir(Hyper.processed_content_folder))
        with open(Hyper.sorted_docs) as fd:
            for line in tqdm.tqdm(fd):
                _, file = line.strip().split('\t')

                if file not in processed_files:
                    if split:
                        self.titles.append(EMPTY_TITLE)
                        self.corpus.append(EMPTY_DOC)
                    else:
                        self.titles.append(EMPTY_TITLE[0])
                        self.corpus.append(EMPTY_DOC[0])
                    continue

                with open(os.path.join(Hyper.processed_content_folder, file)) as doc_fd:
                    title, doc = doc_fd.read().strip().split(TITLE_SEP)

                    if split:
                        self.titles.append(title.split(' '))
                        self.corpus.append(doc.split(' '))
                    else:
                        self.titles.append(title)
                        self.corpus.append(doc)

    def process(self):
        with open(Hyper.processed_queries) as fd:
            queries = fd.readlines()

        query_to_docs = collections.defaultdict(list)
        with open(Hyper.sample_submission) as fd:
            for line in fd:
                line = line.strip().split(',')
                query_to_docs[line[0]].append(line[1])

        model_results = ModelResult([])
        for query in tqdm.tqdm(queries):
            query_id, query = query.strip().split('\t')
            query_result = QueryResult(int(query_id), [])

            for doc_id in query_to_docs[query_id]:
                doc_result = DocScore(doc_id, self.get_score(query, int(doc_id) - 1))
                query_result.results.append(doc_result)

            model_results.queries.append(query_result)

        print('save results to {}'.format(self.results_folder))
        with open(self.results_folder, 'w') as fd:
            for query in model_results.queries:
                for doc in query.results:
                    fd.write('{}\t{}\t{}\n'.format(query.id, doc.id, doc.score))

        return model_results

    def get_score(self, query, doc_id):
        raise NotImplemented


class BM25(BaseModel):
    results_folder = Hyper.bm25_result

    def __init__(self, corpus):
        super(BM25, self).__init__()

        self.corpus = corpus
        self.bm25 = gensim_bm25(corpus)

    def get_score(self, query, doc_id):
        return self.bm25.get_score(query.split(' '), int(doc_id))


class YandexModel(BaseModel):
    results_folder = Hyper.yandex_result

    def __init__(self):
        super(YandexModel, self).__init__()

        self.cf = collections.defaultdict(int)
        self.couples_in_direct_order = {}
        self.couples_in_reverse_order = {}
        self.couples_through_one = {}
        self.counter = {}
        self.avgdl = 0.0

        self.load_corpus()

        self._initialize()
        self._initialize_pairs()

    def _initialize(self):
        for i, doc in tqdm.tqdm(enumerate(self.corpus)):
            self.counter[i] = collections.defaultdict(int)

            for word in doc:
                self.counter[i][word] += 1
                self.counter[i][TOTAL_WORDS_IN_DOC] += 1

                self.cf[word] += 1
                self.cf[TOTAL_WORDS_IN_CORPUS] += 1
                self.avgdl += 1

            for word in self.titles[i]:
                self.counter[i][word] += 1
                self.counter[i][TOTAL_WORDS_IN_DOC] += 1

                self.cf[word] += 1
                self.cf[TOTAL_WORDS_IN_CORPUS] += 1
                self.avgdl += 1

        self.avgdl /= len(self.corpus)

    def _initialize_pairs(self):
        for i, doc in tqdm.tqdm(enumerate(self.corpus)):
            self.couples_in_direct_order[i] = collections.defaultdict(int)
            self.couples_in_reverse_order[i] = collections.defaultdict(int)
            self.couples_through_one[i] = collections.defaultdict(int)

            prev_word, prev_prev_word = None, None
            for word in doc:
                if prev_word is not None:
                    self.couples_in_direct_order[i][(prev_word, word)] += 1
                    self.couples_in_reverse_order[i][(word, prev_word)] += 1

                    if prev_prev_word is not None:
                        self.couples_through_one[i][(prev_prev_word, word)] += 1

                    prev_prev_word = prev_word
                prev_word = word

            prev_word, prev_prev_word = None, None
            for word in self.titles[i]:
                if prev_word is not None:
                    self.couples_in_direct_order[i][(prev_word, word)] += 1
                    self.couples_in_reverse_order[i][(word, prev_word)] += 1

                    if prev_prev_word is not None:
                        self.couples_through_one[i][(prev_prev_word, word)] += 1

                    prev_prev_word = prev_word
                prev_word = word

    def get_score(self, query, doc_id):
        return self._get_single_score(query, doc_id) + self._get_pairs_score(query, doc_id)
    
    def _get_single_score(self, query, doc_id):
        score = 0.0
        for word in query.split(' '):
            if word not in self.counter[doc_id]:
                continue

            icf = self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[word]
            tf = self.counter[doc_id][word]

            score += math.log(icf) * tf / (tf + 1 + self.counter[doc_id][TOTAL_WORDS_IN_DOC] / self.avgdl)

            if word in self.titles[doc_id]:
                hdr = 1.
                score += math.log(icf) * 0.2 * hdr / (1 + hdr)

        return score

    def _get_pairs_score(self, query, doc_id):
        score = 0.0

        prev_word = None
        for word in query.split(' '):
            if prev_word is None:
                prev_word = word
                continue

            if (
                (prev_word, word) not in self.couples_in_direct_order[doc_id] and
                (prev_word, word) not in self.couples_through_one[doc_id] and
                (prev_word, word) not in self.couples_in_reverse_order[doc_id]
            ):
                prev_word = word
                continue

            tf = (
                self.couples_in_direct_order[doc_id][(prev_word, word)] +
                0.5 * self.couples_in_reverse_order[doc_id][(prev_word, word)] +
                0.5 * self.couples_through_one[doc_id][(prev_word, word)]
            )

            icf_1 = self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[word]
            icf_2 = self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[prev_word]

            score += 0.3 * (math.log(icf_1) + math.log(icf_2)) * tf / (1 + tf)

            prev_word = word

        prev_word, prev_prev_word = None, None
        for word in query.split(' '):
            if prev_word is None:
                prev_word = word
                continue

            if prev_prev_word is None:
                prev_prev_word, prev_word = prev_word, word
                continue

            if (prev_prev_word, word) not in self.couples_in_direct_order[doc_id]:
                prev_prev_word, prev_word = prev_word, word
                continue

            tf = 0.1 * self.couples_in_direct_order[doc_id][(prev_prev_word, word)]

            icf_1 = self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[word]
            icf_2 = self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[prev_prev_word]

            score += 0.3 * (math.log(icf_1) + math.log(icf_2)) * tf / (1 + tf)

            prev_prev_word, prev_word = prev_word, word

        return score

    def _get_all_words_score(self, query, doc_id):
        n_miss = 0
        score = 0.0
        for word in query.split(' '):
            if word not in self.counter[doc_id]:
                n_miss += 1
                continue

            score += math.log(self.cf[TOTAL_WORDS_IN_CORPUS] / self.cf[word])

        return score * (0.03**n_miss)

    def _get_phrase_score(self, query, doc_id):
        pass

class Lsa(BaseModel):
    results_folder = Hyper.lsa_result

    def __init__(self, n_components=300):
        super(Lsa, self).__init__()

        self.svd = TruncatedSVD(n_components, random_state=42)
        self.vectorizer = CountVectorizer()

    @staticmethod
    def load():
        with open(Hyper.lsa_pickle, 'rb') as fd:
            return pickle.load(fd)

    def save(self):
        with open(Hyper.lsa_pickle, 'wb') as fd:
            pickle.dump(self, fd, protocol=4)

    def fit(self):
        print('create counter')
        new_corpus = self.vectorizer.fit_transform(self.corpus)

        print('fit lsa')
        self.svd.fit(new_corpus)

        print('save lsa')
        self.save()

    def process(self):
        with open(Hyper.processed_queries) as fd:
            queries = fd.readlines()

        query_to_docs = collections.defaultdict(list)
        with open(Hyper.sample_submission) as fd:
            fd.readline()
            for line in fd:
                line = line.strip().split(',')
                query_to_docs[int(line[0]) - 1].append(int(line[1]) - 1)

        queries_vec = list(map(lambda x: x.strip().split('\t')[1], queries))
        queries_vec = self.vectorizer.transform(queries_vec)
        corpus_vec = self.vectorizer.transform(self.corpus)

        queries_vec = self.svd.transform(queries_vec)
        corpus_vec = self.svd.transform(corpus_vec)

        model_results = ModelResult([])
        for query_id, doc_ids in tqdm.tqdm(query_to_docs.items()):
            sim = cosine_similarity(queries_vec[query_id].reshape(1, -1), corpus_vec[doc_ids])

            query_result = QueryResult(int(query_id) + 1, [])
            for i, doc_id in enumerate(doc_ids):
                doc_result = DocScore(doc_id + 1, sim[0][i])
                query_result.results.append(doc_result)

            model_results.queries.append(query_result)

        print('save results to {}'.format(self.results_folder))
        with open(self.results_folder, 'w') as fd:
            for query in model_results.queries:
                for doc in query.results:
                    fd.write('{}\t{}\t{}\n'.format(query.id, doc.id, doc.score))

        return model_results
