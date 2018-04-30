from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np


class GridSearch:

    def predict(self, features, labels, window_size):
        sig_len = (len(features) + 1) * window_size
        test_index = int(int(sig_len / 5 * 4) / window_size)
        X_train = features[:test_index]
        X_test = features[test_index:]
        Y_train = labels[:test_index]
        Y_test = labels[test_index:]
        classifier = KNeighborsClassifier()

        parameters = {
            'n_neighbors': [ 11],
            'weights': [ 'distance'],
            'p': [ 2]
        }

        grid_search = GridSearchCV(classifier,
                                   parameters,
                                   scoring=metrics.make_scorer(metrics.accuracy_score),
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=10)
        grid_search.fit(X_train, Y_train)
        self.predicted = grid_search.predict(X_test)
        return self.confusion_matrix(Y_test, self.predicted)

    def confusion_matrix(self, Y_test, Y_predicted):
        confusion = metrics.confusion_matrix(Y_test, Y_predicted)
        return confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]

    def get_peaks(self, predicted_regions, window_size, signal):
        siglen = len(signal) + 1
        test_index = int(siglen/5)*4
        Y_predicted = []
        window_start = test_index
        for label in predicted_regions:
            if label == 1:
                window_end = window_start + window_size
                qrs_region = [abs(signal[value]) for value in range(window_start, window_end)]
                peak = window_start + np.argmax(qrs_region)
                Y_predicted.append(peak)
            window_start += window_size
        return Y_predicted
