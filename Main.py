import wfdb
import numpy as np
from FeatureExtraction import FeatureExtraction
from KNN import KNN
from GridSearch import GridSearch
from Utility import Utility
from Validation import Validation
import os

fe = FeatureExtraction()
ut = Utility()
val = Validation()

for name in os.listdir("features"):
    signal_name = name.replace(".tsv","")
    train_features, train_labels = fe.extract_features("sample/"+signal_name)
    GridSearch(signal_name, train_features, train_labels, ["not QRS", "QRS"])


