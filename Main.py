import wfdb
import numpy as np
from FeatureExtraction import FeatureExtraction
from KNN import KNN
from GridSearch import GridSearch
from Utility import Utility
from Validation import Validation

fe = FeatureExtraction()
ut = Utility()
val = Validation()

train_features, train_labels = fe.extract_from_all()
val.validate_for_all(train_features, train_labels)
#GridSearch(train_features, train_labels, ["not QRS", "QRS"])'''


