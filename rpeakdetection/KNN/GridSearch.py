from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pickle
import time

class GridSearch:

    def predict(self, X_train, X_test, y_train, name):

        classifier = KNeighborsClassifier()
        parameters = {
            'n_neighbors': np.arange(1,21, 2),
            'weights': ['uniform','distance'],
        }

        grid_search = GridSearchCV(classifier,
                                   parameters,
                                   scoring=metrics.make_scorer(metrics.f1_score),
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)
        print("training")
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
        with open('rpeakdetection/KNN/classifiers/knn_'+name+'.pkl', 'wb') as fid:
            pickle.dump(best_classifier, fid)
        start_time = time.time()
        predicted = grid_search.predict(X_test)
        return start_time, predicted


