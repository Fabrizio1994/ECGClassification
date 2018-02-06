from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pprint as pp


class GridSearch:
    def __init__(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)
        kNN_classifier = KNeighborsClassifier()
        kMeans_classifier = KMeans(n_clusters=2)
        # With a Pipeline object we can assemble several steps
        # that can be cross-validated together while setting different parameters.
        pipelines = [kNN_classifier, kMeans_classifier]

        parameters_kNN = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        parameters = [parameters_kNN, kMeans_classifier.get_params()]
        ## Create a Grid-Search-Cross-Validation object
        ## to find in an automated fashion the best combination of parameters

        for i in range(len(parameters)):
            grid_search = GridSearchCV(pipelines[i],
                                       parameters[i],
                                       scoring=metrics.make_scorer(metrics.f1_score, average='weighted'),
                                       cv=5,
                                       n_jobs=-1,
                                       verbose=10)
            ## Start an exhaustive search to find the best combination of parameters
            ## according to the selected scoring-function

            grid_search.fit(X_train, y_train)  ## Print results for each combination of parameters.
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

            # Let's train the classifier that achieved the best performance,
            # considering the select scoring-function,
            # on the entire original TRAINING-Set
            Y_predicted = grid_search.predict(X_test)
            target_names = ['peak', 'not peak']
            # Evaluate the performance of the classifier on the original Test-Set
            output_classification_report = metrics.classification_report(
                y_test,
                Y_predicted,
                target_names=target_names)

            print("----------------------------------------------------")
            print(output_classification_report)
            print("----------------------------------------------------")

            # Compute the confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test, Y_predicted)

            print("Confusion Matrix: True-Classes X Predicted-Classes")
            print(confusion_matrix)

            # Compute the Normalized-accuracy
            normalized_accuracy = metrics.accuracy_score(y_test, Y_predicted)

            print("Normalized Accuracy: ")
            print(normalized_accuracy)

            # Compute the Matthews Corrcoef value
            f1_score = metrics.f1_score(y_test, Y_predicted)

            print("f1 score: ")
            print(f1_score)



