from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pprint as pp


############################################

class GridSearch():
    def __init__(self, classifier, params, X, target, target_names):

        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.20)
        ## Create a Grid-Search-Cross-Validation object
        ## to find in an automated fashion the best combination of parameters.
        grid_search = GridSearchCV(classifier,
                                   params,
                                   scoring=metrics.make_scorer(metrics.f1_score, average='weighted'),
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=10)

        ## Start an exhaustive search to find the best combination of parameters
        ## according to the selected scoring-function.
        print
        grid_search.fit(X_train, y_train)
        print

        ## Print results for each combination of parameters.
        number_of_candidates = len(grid_search.cv_results_['params'])
        print("Results:")
        for i in range(number_of_candidates):
            print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
                  (grid_search.cv_results_['params'][i],
                   grid_search.cv_results_['mean_test_score'][i],
                   grid_search.cv_results_['std_test_score'][i]))
        print
        print("Best Parameters:")
        pp.pprint(grid_search.best_params_)
        print
        print("Used Scorer Function:")
        pp.pprint(grid_search.scorer_)
        print
        print("Number of Folds:")
        pp.pprint(grid_search.n_splits_)
        print

        # Let's train the classifier that achieved the best performance,
        # considering the select scoring-function,
        # on the entire original TRAINING-Set
        Y_predicted = grid_search.predict(X_test)

        # Evaluate the performance of the classifier on the original Test-Set
        output_classification_report = metrics.classification_report(
            y_test,
            Y_predicted,
            target_names=target_names)
        print
        print
        "----------------------------------------------------"
        print(output_classification_report)
        print
        "----------------------------------------------------"
        print

        # Compute the confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_test, Y_predicted)
        print
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)