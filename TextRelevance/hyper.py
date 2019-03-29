# coding: utf-8


class Hyper(object):
    content_folder = '/home/pmankevich/git/TextRelevance/data/content/20190128'
    processed_content_folder = '/home/pmankevich/git/TextRelevance/processed_data/content'
    n_processes = 16

    doc_urls = '/home/pmankevich/git/TextRelevance/data/urls.numerate.txt'
    sorted_docs = '/home/pmankevich/git/TextRelevance/processed_data/sorted_docs.txt'

    # can be replaced by base queries.txt
    spellchecked_queries = '/home/pmankevich/git/TextRelevance/processed_data/spellchecked_queries.txt'

    queries = '/home/pmankevich/git/TextRelevance/data/queries.numerate.txt'
    processed_queries = '/home/pmankevich/git/TextRelevance/processed_data/queries.numerate.txt'

    sample_submission = '/home/pmankevich/git/TextRelevance/data/sample_submission.txt'
    submission = '/home/pmankevich/git/TextRelevance/processed_data/submission.txt'

    base_results = '/home/pmankevich/git/TextRelevance/results/base_results.txt'
    bm25_result = '/home/pmankevich/git/TextRelevance/results/bm25.txt'
    yandex_result = '/home/pmankevich/git/TextRelevance/results/yandex_result.txt'
    lsa_result = '/home/pmankevich/git/TextRelevance/results/lsa_result.txt'
    lda_result = '/home/pmankevich/git/TextRelevance/results/lda_result.txt'

    lsa_pickle = '/home/pmankevich/git/TextRelevance/results/lsa.pkl'
    lda_pickle = '/home/pmankevich/git/TextRelevance/results/lda.pkl'

    # always equals one of model results
    current_model_result = yandex_result
