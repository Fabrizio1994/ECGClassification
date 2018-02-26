import wfdb
import numpy as np
from FeatureExtraction import FeatureExtraction
from KNN import KNN
from GridSearch import GridSearch

fe = FeatureExtraction()


train_features, train_labels = fe.extract_features('sample/100')
test_features, test_labels = fe.extract_features('sample/215')
knn = KNN(train_features, train_labels, test_features, test_labels)
#GridSearch(train_features, train_labels, test_features, test_labels, ["not QRS", "QRS"])


