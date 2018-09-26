from beatclassification.SVM_weighted.FeatureExtraction import FeatureExtraction
from rpeakdetection.Utility import Utility
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as splt
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import itertools
import pickle
ut = Utility()
fe = FeatureExtraction()



train_dataset = ["101", "106", "108","109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203",
                 "205", "207", "208", "209", "215", "220", "223", "230"]
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
ann_path = 'data/ecg/mitdb/'

group_names = ['rr', 'hos', 'raw', 'wavelets']
group_comb = list()
for i in range(len(group_names)):
    for comb in itertools.combinations(group_names, i):
        if len(comb) > 0:
            group_comb.append(comb)



def score_func(y_true, pred):
    return np.mean(metrics.f1_score(y_true, pred, average=None))


scale_factors = [-14, 4, 1, 10]
max = 0
best_group = None
best_precision = 0
best_recall = 0
best_f1 = 0
Y_test = None
best_pred = None
best_model = None
peaks = ut.remove_non_beat_for_all(ann_path)[0]
for feat_group in group_comb:
    print(feat_group)
    X_train, Y_train = fe.extract(train_dataset,features_group=feat_group, ann_path=ann_path, peaks=peaks,
                                  scale_factors=scale_factors,from_annot=True)
    train = Y_train.tolist()
    weights = {q: 1/train.count(q) for q in range(0,4)}
    X_test, Y_test = fe.extract(test_dataset, features_group=feat_group, ann_path=ann_path, peaks=peaks, from_annot=True)
    random_state = np.random.seed(42)
    classifier = LinearSVC(class_weight='balanced')
    params = {
        'base_estimator' : [None, classifier],
        'max_samples' : np.arange(0.5, 1.1, 0.1),
        'max_features': np.arange(0.5, 1.1, 0.1)
    }
    grid_search = GridSearchCV(BaggingClassifier(), params, scoring=metrics.make_scorer(score_func),
                           cv=5, n_jobs=-1, verbose=10)
    grid_search.fit(X_train, Y_train)
    model = grid_search.best_estimator_
    print(model)
    predicted = grid_search.predict(X_test)
    splt.metrics.plot_confusion_matrix(Y_test, predicted)
    precision = precision_score(Y_test, predicted, average=None)
    recall = recall_score(Y_test, predicted, average=None)
    f1 = f1_score(Y_test, predicted, average=None)
    print(precision)
    print(recall)
    if np.mean(f1) > max:
        max = np.mean(f1)
        best_model = grid_search.best_estimator_
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_group = feat_group
        best_pred = predicted
    print("best: model, precision, recall, f1, group")
    print(best_model)
    print(best_precision)
    print(best_recall)
    print(best_f1)
    print(best_group)
splt.metrics.plot_confusion_matrix(Y_test, best_pred)
print("best: model, precision, recall, f1, group")
print(best_model)
print(best_precision)
print(best_recall)
print(best_f1)
print(best_group)
with open('ensemble.pkl', 'wb') as fid:
    pickle.dump(best_model, fid)
plt.show()







