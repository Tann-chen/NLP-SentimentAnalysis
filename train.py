from nlp import read_file
from nlp import process_document
from nlp import word_to_vect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import sklearn.metrics
from sklearn.decomposition import PCA
import gensim

import os
import sys
import time


def get_training_data(pos_corpus, neg_corpus, feature_select):
    X = []
    Y = []
    # postive data
    for file_name in os.listdir(pos_corpus):
        if file_name.startswith('.'):
            continue
        docu = read_file(pos_corpus + '/' + file_name)
        processed = ''
        if feature_select == 'w2v':
            processed = process_document(docu, False)
        else:
            processed = process_document(docu, True)

        if len(processed) > 0:
            X.append(processed)
            Y.append(1)  # 1 means pos

    pos_num = len(Y)

    # negative
    for file_name in os.listdir(neg_corpus):
        if file_name.startswith('.'):
            continue
        docu = read_file(neg_corpus + '/' + file_name)
        processed = ''
        if feature_select == 'w2v':
            processed = process_document(docu, False)
        else:
            processed = process_document(docu, True)

        if len(processed) > 0:
            X.append(processed)
            Y.append(0)  # 0 means neg

    print("\n-------- Courpus Infos --------\n")
    print("Postive data amount : " + str(pos_num))
    print("Negative data amount : " + str(len(Y) - pos_num))
    print("\n")
    print("[INFO] all data have loaded and processed.")
    return X, Y


def train_by_TO(X, Y):
    print("[INFO] training TO classifier ...")
    TO = CountVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 1))
    svm_algo = svm.LinearSVC(C=1.0)
    classifier = Pipeline([('vectorizer', TO), ('pac', svm_algo)])
    classifier.fit(X, Y)
    joblib.dump(classifier, "svm_TO_classifier.pkl")
    print("[INFO] ^_^ finish training TO classifier.")
    del classifier


def train_by_tfidf(X, Y):
    print("[INFO] training svm-tfidf classifier ...")
    tf_idf = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf = True, use_idf = True, ngram_range=(1, 1))
    svm_algo = svm.LinearSVC(C=1.0)
    classifier = Pipeline([('vectorizer', tf_idf), ('pac', svm_algo)])
    classifier.fit(X, Y)
    joblib.dump(classifier, "svm_tfidf_classifier.pkl")
    print("[INFO] ^_^ finish training svm-tfidf classifier.")
    del classifier


def train_by_word2vec(X, Y):
    print("[INFO] word2vec model is loading ...")
    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)
    print("[INFO] word2vec model has been loaded.")
    print("[INFO] convert sentences to vectors ...")

    X_vectors = []
    for s in X:
        tokens = s.split()
        sentence_vec = []
        for i in range(0, 300):
            sentence_vec.append(0)

        for t in tokens:
            vec = word_to_vect(t, model)
            if vec is not None:
                for i in range(0, 300):
                    sentence_vec[i] = sentence_vec[i] + vec[i]

        for i in range(0, 300):
            sentence_vec[i] = sentence_vec[i] / len(tokens)

        X_vectors.append(sentence_vec)

    print("[INFO] training svm-word2vec classifier ...")
    classifier = svm.LinearSVC(C = 1.0)
    classifier.fit(X_vectors, Y)
    joblib.dump(classifier, "svm_word2vec_classifier.pkl")
    print("[INFO] ^_^ finish training svm-word2vec classifier.")
    del classifier


if __name__ == "__main__":
    pos_corpus_train = "reviews/train_pos"
    neg_corpus_train = "reviews/train_neg"

    start_time = time.time()
    X_train, Y_train = get_training_data(pos_corpus_train, neg_corpus_train, 'to')
    train_by_TO(X_train, Y_train)
    end_time = time.time()
    print("[INFO] running time : " + str(end_time - start_time))
    print("-----------------")

    del X_train
    del Y_train

    start_time = time.time()
    X_train, Y_train = get_training_data(pos_corpus_train, neg_corpus_train, 'tfidf')
    train_by_tfidf(X_train, Y_train)
    end_time = time.time()
    print("[INFO] running time : " + str(end_time - start_time))
    print("-----------------")

    del X_train
    del Y_train

    start_time = time.time()
    X_train, Y_train = get_training_data(pos_corpus_train, neg_corpus_train, 'w2v')
    train_by_word2vec(X_train, Y_train)
    end_time = time.time()
    print("[INFO] running time : " + str(end_time - start_time))
