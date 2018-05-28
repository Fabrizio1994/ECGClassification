from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer


class SVCGridSearch():
    def __init__(self, X_train, y_train, X_test):
        params = {
            'class_weight' : ['balanced'],
            'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        }
        scorer = make_scorer(self.score_func)
        classifier = LinearSVC()
        grid_search = GridSearchCV(classifier,
                                   params,
                                   scoring=scorer,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=10)
        grid_search.fit(X_train, y_train)


        ## Print results for each combination of parameters.
        number_of_candidates = len(grid_search.cv_results_['params'])
        print("Results:")
        for i in range(number_of_candidates):
            print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
                  (grid_search.cv_results_['params'][i],
                   grid_search.cv_results_['mean_test_score'][i],
                   grid_search.cv_results_['std_test_score'][i]))

        print(grid_search.best_estimator_)
        self.y_predicted = grid_search.predict(X_test)

    # TODO: try to optimize this function
    def score_func(self, y_true, y_pred):
        weights = []
        for label in y_true:
            weights.append(1 / list(y_true).count(label))
        acc = metrics.accuracy_score(y_true, y_pred, sample_weight=weights)
        return acc



