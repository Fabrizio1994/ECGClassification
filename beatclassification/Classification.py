from beatclassification.Preprocessing import Preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import scikitplot as splt
import matplotlib.pyplot as plt
from beatclassification.data_visualization import data_visualization

prep = Preprocessing()
dv = data_visualization()
train_shape = (51011, 170)
test_shape = (49701, 170)


def score_func(y_true, pred):
    return np.mean(metrics.f1_score(y_true, pred, average=None))


train_dataset = ['106', '112', '122', '201', '223', '230', "108", "109", "115", "116", "118", "119", "124",
                 "205", "207", "208", "209", "215", '101', '114', '203', '220']
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
distribution = dv.data_distribution(train_dataset, aami=True)
X_train, Y_train = prep.preprocess(train_dataset, train_shape, one_hot=False)
augment_F = distribution['V']/distribution['F']
X_train, Y_train = prep.augment_data(X_train, Y_train, ['N', 'S','V', 'F'], label='S', factor=2, one_hot=False)
X_train, Y_train = prep.augment_data(X_train, Y_train, ['N', 'S','V', 'F'], label='F', factor=2, one_hot=False)
X_test, Y_test = prep.preprocess(test_dataset, test_shape, one_hot=False)
classifier = KNeighborsClassifier()
parameters = {
    'n_neighbors': np.arange(1, 21, 2),
    'weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(classifier,
                           parameters,
                           scoring=metrics.make_scorer(score_func),
                           cv=5,
                           n_jobs=-1,
                           verbose=20)
print("training")
grid_search.fit(X_train, Y_train)
model = grid_search.best_estimator_
predicted = model.predict(X_test)
splt.metrics.plot_confusion_matrix(Y_test, predicted)
print("per class precision")
precision = precision_score(Y_test, predicted, average=None)
print(precision)
print("average precision")
print(np.mean(precision))
print('per class recall')
recall = recall_score(Y_test, predicted, average=None)
print(recall)
print('average recall')
print(np.mean(recall))
plt.show()
plt.close()
