from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import LinearSVC


############################################

class SVCGridSearch():
    def __init__(self, X_train, y_train, X_test):
        svc = LinearSVC()

        params = {
            'C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
            'class_weight': ['balanced']
        }

        classifier = svc
        grid_search = GridSearchCV(classifier,
                                   params,
                                   scoring="recall_weighted",
                                   cv=22,
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
        # Let's train the classifier that achieved the best performance,
        # considering the select scoring-function,
        # on the entire original TRAINING-Set

        self.y_predicted = grid_search.predict(X_test)




