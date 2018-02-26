import wfdb
import numpy as np
from FeatureExtraction import FeatureExtraction
from KNN import KNN
from GridSearch import GridSearch
from Utility import Utility

fe = FeatureExtraction()
ut = Utility()



#train_features, train_labels = fe.extract_features('sample/100')
#test_features, test_labels = fe.extract_features('sample/215')

test_features, test_labels = ut.extract_features('features/100.tsv')


#knn = KNN(train_features, train_labels, test_features, test_labels)
#GridSearch(train_features, train_labels, test_features, test_labels, ["not QRS", "QRS"])


