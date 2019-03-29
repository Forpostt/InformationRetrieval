# coding: utf-8

import collections
import operator
from hyper import Hyper


def create_submission():
    print('Warning!!! Current results: {}'.format(Hyper.current_model_result))

    results = collections.defaultdict(list)
    with open(Hyper.current_model_result) as fd:
        for line in fd:
            query_id, doc_id, score = line.strip().split('\t')
            results[int(query_id)].append((int(doc_id), float(score)))

    results = sorted(results.items(), key=operator.itemgetter(0))
    with open(Hyper.submission, 'w') as fd:
        fd.write('QueryId,DocumentId\n')
        for query_id, docs in results:
            docs = sorted(docs, key=operator.itemgetter(1), reverse=True)
            for doc_id, _ in docs:
                fd.write('{},{}\n'.format(query_id, doc_id))


if __name__ == '__main__':
    create_submission()
