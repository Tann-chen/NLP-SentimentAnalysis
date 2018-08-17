import os
from train import get_training_data
from nlp import word_to_vect
import gensim
import sklearn.metrics
from sklearn.externals import joblib
from sklearn.decomposition import PCA



def test_TO(X_test, Y_test):
    print("\n -----------  SVM TO REPORT  ----------- \n")
    svm_TO_classifier = joblib.load('svm_TO_classifier.pkl')
    Y_pred = svm_TO_classifier.predict(X_test)
    print("accuracy_score: " + str(sklearn.metrics.accuracy_score(Y_pred, Y_test)))
    del Y_pred


def test_tfidf(X_test, Y_test):
    print("\n -----------  SVM tf-idf REPORT  ----------- \n")
    svm_tfidf_classifier = joblib.load('svm_tfidf_classifier.pkl')
    Y_pred = svm_tfidf_classifier.predict(X_test)
    print("accuracy_score: " + str(sklearn.metrics.accuracy_score(Y_pred, Y_test)))
    del Y_pred


def test_word2vec(X_test, Y_test):
    print("[INFO] word2vec model is loading ...")
    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)
    print("[INFO] word2vec model has been loaded.")
    print("[INFO] convert sentences to vectors ...")
    X_vectors = []
    for s in X_test:
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

    print("[INFO] reduce paramters ...")
    svm_w2v_classifier = joblib.load('svm_word2vec_classifier.pkl')
    print("\n -----------  SVM word2vec REPORT  ----------- \n")
    Y_pred = svm_w2v_classifier.predict(X_vectors)
    print("accuracy_score: " + str(sklearn.metrics.accuracy_score(Y_pred, Y_test)))


if __name__ == "__main__":
    pos_corpus_test = "reviews/test_pos"
    neg_corpus_test = "reviews/test_neg"
    X_test, Y_test = get_training_data(pos_corpus_test, neg_corpus_test, 'tfidf')
    X_test1, Y_test1 = get_training_data(pos_corpus_test, neg_corpus_test, 'w2v')

    test_TO(X_test, Y_test)
    test_tfidf(X_test, Y_test)
    test_word2vec(X_test1, Y_test1)
