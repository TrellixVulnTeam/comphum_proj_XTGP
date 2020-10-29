import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model


class Model:
    def __init__(self, term_dict):
        self.term_dict = term_dict

    def make_data(self):
        text, labels = [], []
        for term, doc_passages in self.term_dict.items():
            for doc_id, passages in doc_passages.items():
                for passage in passages:
                    text.append(passage)
                    labels.append(term)
        assert len(text) == len(labels)
        return text, labels

    def process_data(self):
        corpus, labels = self.make_data()
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(corpus)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        y = self.le.transform(labels)
        return X, y

    def run_model(self):
        X, y = self.process_data()
        clf = linear_model.LogisticRegression(random_state=0).fit(X, y)
        # predictive features
        for ix, arr in enumerate(clf.coef_):
            max_indices = reversed(arr.argsort()[-20:])
            for maxi in max_indices:
                label = self.vectorizer.get_feature_names()[maxi]
                print("{} : {}".format(label, arr[maxi]))
            print()
            min_indices = arr.argsort()[:20]
            for mini in min_indices:
                label = self.vectorizer.get_feature_names()[mini]
                print("{} : {}".format(label, arr[mini]))




