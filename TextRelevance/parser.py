# coding: utf-8

import re
import nltk
from inscriptis import get_text
from pymystem3 import Mystem

TITLE_SEP = '<TITLE_SEP_PMANKEVICH>'
TITLE_MATCH = re.compile('<title>(.*?)</title>')
NOT_DIGIT_OR_LETTER = re.compile('\W+')
NOT_DIGIT = re.compile('[0-9 ]+')


class Parser(object):
    @staticmethod
    def find_doc_url(doc_body):
        return doc_body.split('\n', 1)

    @staticmethod
    def _parse_html(doc_body):
        title = re.search(TITLE_MATCH, doc_body)
        if title:
            title = title.group(1)
        else:
            title = ''

        doc_body = doc_body.replace('<', ' <').replace('>', '> ')
        return '{}{}{}'.format(title, TITLE_SEP, get_text(doc_body))

    @staticmethod
    def _parse_pdf(doc_body):
        return '{}{}'.format(TITLE_SEP, doc_body)

    @staticmethod
    def _parse(doc):
        doc_url, doc_body = Parser.find_doc_url(doc)
        return Parser._parse_html(doc_body)

    @staticmethod
    def parse(doc_id, doc):
        try:
            return Parser._parse(doc)
        except Exception as e:
            print(doc_id, e)


class Normalizer(object):
    @staticmethod
    def normalize(doc_id, doc):
        try:
            return Normalizer._normalize(doc)
        except Exception as e:
            print(doc_id, e)

    @staticmethod
    def _normalize(doc):
        return re.sub(NOT_DIGIT_OR_LETTER, ' ', doc.replace('_', ' ').lower()).strip()


class DigitNormalizer(Normalizer):
    @staticmethod
    def _normalize(doc):
        res = re.sub(NOT_DIGIT_OR_LETTER, ' ', doc.replace('_', ' ').lower()).strip()
        return re.sub(NOT_DIGIT, ' ', res)


class Lemmatisation(object):
    def __init__(self):
        self.ru_lem = Mystem()
        self.en_lem = nltk.stem.WordNetLemmatizer()

        self.ru_stop_words = set(nltk.corpus.stopwords.words('russian') + [chr(i) for i in range(ord('а'), ord('я') + 1)])
        self.en_stop_words = set(nltk.corpus.stopwords.words('english') + [chr(i) for i in range(ord('a'), ord('z') + 1)])

    def visible(self, term):
        if re.search(NOT_DIGIT_OR_LETTER, term) or term in self.ru_stop_words or term in self.en_stop_words:
            return False
        return True

    def _lemmatize(self, doc):
        lemmas = self.ru_lem.lemmatize(doc)
        lemmas = [self.en_lem.lemmatize(lemma) for lemma in lemmas if self.visible(lemma)]
        return ' '.join(lemmas)

    def lemmatize(self, doc_id, doc):
        try:
            return self._lemmatize(doc)
        except Exception as e:
            print(doc_id, e)
