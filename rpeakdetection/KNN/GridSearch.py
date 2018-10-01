from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pickle

class GridSearch:

    def predict(self, X_train, X_test, y_train, name, test_size):

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
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
        with open('rpeakdetection/KNN/classifiers/knn_'+name+'_'+str(test_size)+'.pkl', 'wb') as fid:
            pickle.dump(best_classifier, fid)
        return grid_search.predict(X_test)


