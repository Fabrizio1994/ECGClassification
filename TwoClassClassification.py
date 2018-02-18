from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch
import numpy as np
from Utility import Utility
import os

fe = FeatureExtraction()
ut = Utility()


class TwoClassClassification:
    def __init__(self, num_signals):
        names = []
        target_names = ["not QRS", "QRS"]
        features = []
        labels = []
        counter = 0
        for file in os.listdir(os.getcwd()+"/samples"):
            if counter < num_signals:
                sig_name = file.split(".")[0]
                if sig_name not in names:
                    print(sig_name)
                    counter += 1
                    names.append(sig_name)
                    feature, label = fe.extract_features(sig_name)
                    #when features are written into files, use the function below
                    #feature, label = ut.load_feature(sig_name)
                    for sample in feature:
                        features.append(sample)
                    for value in label:
                        labels.append(value)
        print("Grid search: training signals:")
        for name in names:
            print(name)
        GridSearch(np.asarray(features), np.asarray(labels), target_names)
