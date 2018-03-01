from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pprint as pp



class GridSearch:
    def __init__(self, signal_name, features, labels, target_names):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)
        classifier = KNeighborsClassifier()
        # With a Pipeline object we can assemble several steps
        # that can be cross-validated together while setting different parameters.
        parameters = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3]
        }
        ## Create a Grid-Search-Cross-Validation object
        ## to find in an automated fashion the best combination of parameters


        grid_search = GridSearchCV(classifier,
                                   parameters,
                                   scoring=metrics.make_scorer(metrics.accuracy_score),
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
        text_file = open("LOW_Classification_report.txt", "a")
        text_file.write("%s\n"%(signal_name))
        for param in grid_search.best_params_:
            text_file.write("%s\n" % grid_search.best_params_[param])
        text_file.close()
        print("Used Scorer Function:")
        pp.pprint(grid_search.scorer_)

        print("Number of Folds:")
        pp.pprint(grid_search.n_splits_)

        # Let's train the classifier that achieved the best performance,
        # considering the select scoring-function,
        # on the entire original TRAINING-Set
        Y_predicted = grid_search.predict(X_test)
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
        self.classification_report_txt(output_classification_report, confusion_matrix,
                                       normalized_accuracy)

    def classification_report_txt(self, report, confusion_matrix, accuracy):
        text_file = open("LOW_Classification_report.txt", "a")
        text_file.write("report:%s\nconfusion matrix: %s\nnormalized accuracy: %s\n" % (report,
                        confusion_matrix, accuracy))
        text_file.close()



