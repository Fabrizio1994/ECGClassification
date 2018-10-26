from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pickle
import time

class GridSearch:

    def predict(self, X_train, X_test, y_train, name, comb):
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
        classifiers_path = 'rpeakdetection/KNN/classifiers/'
        comb_name = comb[0] + '_' + comb[1] + '_' + comb[2]
        class_name = classifiers_path + name + '_' + comb_name + '.pkl'
        with open(class_name, 'wb') as fid:
            pickle.dump(best_classifier, fid)
        start_time = time.time()
        predicted = grid_search.predict(X_test)
        return start_time, predicted

    def SSK_train(self, X_train, y_train, comb):
        classifier = KNeighborsClassifier()
        parameters = {
            'n_neighbors': np.arange(1, 11, 2),
            'p': [1, 2, 3],
        }
        grid_search = GridSearchCV(classifier,
                                   parameters,
                                   scoring=metrics.make_scorer(metrics.accuracy_score),
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=0)
        print("training")
        grid_search.fit(X_train, y_train)
        class_name = 'rpeakdetection/KNN/classifiers/SSK_'+comb[1]+'_'+comb[2]+'.pkl'
        best_classifier = grid_search.best_estimator_
        with open(class_name, 'wb') as fid:
            pickle.dump(best_classifier, fid)
        return best_classifier

