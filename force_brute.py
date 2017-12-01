# -*- coding: utf-8 -*-

import sys
import json
import logging
from gensim import corpora, models, similarities

from load_corpus import load, cleanDoc, load_unprocessed_documents
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Resources:
    @staticmethod
    def generate_resources():
        global_documents = load()
        documents = global_documents
        data_dictionary = corpora.Dictionary(documents.values())
        data_dictionary.save('path_pre_process/dictionary.dict')
        corpus = [data_dictionary.doc2bow(d) for d in documents.values()]
        corpora.MmCorpus.serialize('path_pre_process/corpora.mm', corpus)

    @staticmethod
    def get_unprocessed_documents():
        return load_unprocessed_documents()


class BruteForce:

    def precalculate(self):
        # Add Before make anything
        # Resources.generate_resources()
        corpus = corpora.MmCorpus('path_pre_process/corpora.mm')
        data_dictionary = corpora.Dictionary.load('path_pre_process/dictionary.dict')
        tfidf = models.TfidfModel(corpus)
        tfidf.save('path_pre_process/tf-idf')
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(data_dictionary))
        index.save('path_pre_process/matrixSimilarities.txt')

    def init_data(self):
        self.dictionary = corpora.Dictionary.load('path_pre_process/dictionary.dict')
        self.tfidf = models.TfidfModel.load('path_pre_process/tf-idf')
        self.index = similarities.SparseMatrixSimilarity.load('path_pre_process/matrixSimilarities.txt')

    def make_query(self, query):
        query = cleanDoc(query)
        vec = self.dictionary.doc2bow(query)

        sims = self.index[self.tfidf[vec]]
        results = list(filter(lambda tup: tup[1] > 0.0, list(enumerate(sims))))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        unprocessed_documents = Resources.get_unprocessed_documents()
        brutus_results = []

        for id, value in results:
            result = dict()
            result['similarity'] = value.item()
            result['document'] = list(unprocessed_documents.values())[id]
            brutus_results.append(result)

        return brutus_results


class LDAModel:

    def __init__(self, passes=5):
        self.passes = passes

    def precalculate_lda_model(self):
        self.corpus = corpora.MmCorpus('path_pre_process/corpora.mm')
        self.data_dictionary = corpora.Dictionary.load('path_pre_process/dictionary.dict')
        print('Initializing LDA MODEL')
        lda = models.ldamulticore.LdaMulticore(corpus=self.corpus, id2word=self.data_dictionary,
                                       num_topics=50, passes=self.passes, workers=3)
        lda.save('path_pre_process/lda-model')
        lda.print_topics(10)

    def process_lda_matrix(self):
        self.lda = models.ldamulticore.LdaMulticore.load('path_pre_process/lda-model')
        index = similarities.SparseMatrixSimilarity(
            self.lda[self.corpus], num_features=len(self.data_dictionary))
        index.save('path_pre_process/lda_matrix')

    def precalculate(self):
        self.precalculate_lda_model()
        self.process_lda_matrix()

    def init_data(self):
        self.dictionary = corpora.Dictionary.load('path_pre_process/dictionary.dict')
        self.lda = models.ldamulticore.LdaMulticore.load('path_pre_process/lda-model')
        self.index = similarities.MatrixSimilarity.load(
            'path_pre_process/lda_matrix')

    def make_query(self, query):
        query = cleanDoc(query)
        vec = self.dictionary.doc2bow(query)

        sims = self.index[self.lda[vec]]
        results = list(filter(lambda tup: tup[1] > 0.0, list(enumerate(sims))))
        results = sorted(results, key=lambda x: x[1], reverse=True)

        unprocessed_documents = Resources.get_unprocessed_documents()
        clustered_results = []

        for id, value in results:
            result = dict()
            result['similarity'] = value.item()
            result['document'] = list(unprocessed_documents.values())[id]
            clustered_results.append(result)

        return clustered_results


if __name__ == '__main__':
    args = sys.argv

    if len(sys.argv) < 2:
        print('Not Model Specified')
        sys.exit()

    if sys.argv[1] == 'LDA':
        model = LDAModel()
    elif sys.argv[1] == 'brute':
        model = BruteForce()

    if sys.argv[2] == 'train':
        # Resources.generate_resources()
        model.precalculate()
    else:
        results = model.make_query(str(sys.argv[2]))
        for result in results[:3]:
            print(result)

