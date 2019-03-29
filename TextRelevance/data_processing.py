# coding: utf-8

import os
import tqdm
import operator
from multiprocessing import Pool

from parser import Parser, Normalizer, Lemmatisation, TITLE_SEP, DigitNormalizer
from hyper import Hyper

DEF_DOC_ID = 1


def sort_docs():
    doc_url_to_doc_id = {}
    with open(Hyper.doc_urls) as fd:
        for line in fd:
            doc_id, doc_url = line.strip().split('\t')
            doc_url_to_doc_id[doc_url] = int(doc_id)

    doc_id_to_doc_file = {}

    base_files = os.listdir(Hyper.content_folder)
    for file in tqdm.tqdm(base_files):
        with open(os.path.join(Hyper.content_folder, file)) as fd:
            url = fd.readline()[:-1]
            doc_id_to_doc_file[doc_url_to_doc_id.get(url, DEF_DOC_ID)] = file

    with open(Hyper.sorted_docs, 'w') as fd:
        for i in range(1, len(base_files) + 1):
            if i not in doc_id_to_doc_file:
                fd.write('{}\t{}\n'.format(i, doc_id_to_doc_file[DEF_DOC_ID]))
            else:
                fd.write('{}\t{}\n'.format(i, doc_id_to_doc_file[i]))


def subprocess(doc_ids):
    parser = Parser()
    normalizer = Normalizer()
    lemm = Lemmatisation()

    for i, doc_id in enumerate(tqdm.tqdm(doc_ids)):
        with open(os.path.join(Hyper.content_folder, doc_id)) as fd:
            doc = fd.read()

        parsed_doc = parser.parse(doc_id, doc)
        if parsed_doc is None:
            print('cannot parse {}'.format(doc_id))
            continue

        title, doc = parsed_doc.split(TITLE_SEP)

        if title:
            title = lemm.lemmatize(doc_id, title)
            title = normalizer.normalize(doc_id, title)

        if doc:
            doc = lemm.lemmatize(doc_id, doc)
            doc = normalizer.normalize(doc_id, doc)

        with open(os.path.join(Hyper.processed_content_folder, doc_id), 'w') as fd:
            fd.write('{}{}{}'.format(title, TITLE_SEP, doc))


def process():
    files = os.listdir(Hyper.content_folder)

    files_splits = []
    for i in range(Hyper.n_processes):
        files_splits.append(files[i * int(len(files) / Hyper.n_processes + 1):(i + 1) * int(len(files) / Hyper.n_processes + 1)])

    pool = Pool(Hyper.n_processes)
    pool.map(subprocess, files_splits)


def process_queries():
    normalizer = Normalizer()
    lemm = Lemmatisation()

    processed_queries = {}
    with open(Hyper.spellchecked_queries) as fd:
        for line in tqdm.tqdm(fd):
            query_id, query = line.strip().split('\t', 1)
            processed_queries[int(query_id)] = normalizer.normalize(query_id, lemm.lemmatize(query_id, query))

    with open(Hyper.processed_queries, 'w') as fd:
        processed_queries = sorted(processed_queries.items(), key=operator.itemgetter(0))
        for query_id, query in processed_queries:
            fd.write('{}\t{}\n'.format(query_id, query))


if __name__ == '__main__':
    # process()
    process_queries()
    # sort_docs()
