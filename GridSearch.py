from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pprint as pp



class GridSearch:
    def grid_search(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)
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

        grid_search.fit(X_train, y_train)
        number_of_candidates = len(grid_search.cv_results_['params'])
        print("Results:")
        for j in range(number_of_candidates):
            print(j, 'params - %s; mean - %0.3f; std - %0.3f' %
                  (grid_search.cv_results_['params'][j],
                   grid_search.cv_results_['mean_test_score'][j],
                   grid_search.cv_results_['std_test_score'][j]))

        print("Best Estimator:")
        pp.pprint(grid_search.best_estimator_)

        print("Best Parameters:")
        pp.pprint(grid_search.best_params_)

        print("Used Scorer Function:")
        pp.pprint(grid_search.scorer_)

        print("Number of Folds:")
        pp.pprint(grid_search.n_splits_)

        return grid_search.predict(X_test)






