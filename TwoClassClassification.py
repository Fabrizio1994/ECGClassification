from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch
import numpy as np
import os

fe = FeatureExtraction()


class TwoClassClassification:
    def __init__(self):
        names = []
        target_names = ["not QRS", "QRS"]
        features = []
        labels = []
        for file in os.listdir(os.getcwd()+"/samples"):
            sig_name = file.split(".")[0]
            if sig_name not in names:
                names.append(sig_name)
                feature = fe.extract_features(sig_name)
                label = fe.define_2class_labels(sig_name)
                for sample in feature:
                    features.append(sample)
                for value in label:
                    labels.append(value)
        print(len(labels), len(features))
        GridSearch(np.asarray(features), labels, target_names)
