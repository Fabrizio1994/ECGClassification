from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np


class GridSearch:

    def qrs_gridsearch(self, features, labels):
        Y_test, Y_predicted = self.predict(features, labels)
        return self.qrs_confusion_matrix(Y_test, Y_predicted)

    def rpeak_gridsearch(self, features, labels, Y_test, window_size, signal):
        # array of -1,1 for each region of size = window_size
        predicted_regions = self.predict(features, labels)[1]
        Y_predicted = self.get_peaks(predicted_regions, window_size, signal)
        return Y_predicted

    def predict(self, features, labels):
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.20)
        classifier = KNeighborsClassifier()

        parameters = {
            'n_neighbors': [1, 3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

        grid_search = GridSearchCV(classifier,
                                   parameters,
                                   scoring=metrics.make_scorer(metrics.accuracy_score),
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=10)
        grid_search.fit(X_train, Y_train)
        Y_predicted = grid_search.predict(X_test)
        return Y_test, Y_predicted

    def qrs_confusion_matrix(self, Y_test, Y_predicted):
        confusion = metrics.confusion_matrix(Y_test, Y_predicted)
        return confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]

    def get_peaks(self, predicted_regions, window_size, signal):
        siglen = len(signal) + 1
        test_index = int(siglen/5)*4
        test_size = siglen - test_index
        index = int(test_size / window_size)
        Y_predicted = []
        window_start = test_index
        for label in predicted_regions[-index:]:
            if label == 1:
                window_end = window_start + window_size + 1
                qrs_region = [abs(signal[value]) for value in range(window_start, window_end)]
                peak  = window_start + np.argmax(qrs_region)
                Y_predicted.append(peak)
            window_start += window_size
        return Y_predicted
