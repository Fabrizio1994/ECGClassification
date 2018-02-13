from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch
import os
import numpy as np

fe = FeatureExtraction()


class MultiClassClassification:
    def __init__(self):
        multiclass_target_names = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j'
            , 'n', 'E', '/', 'f', 'Q', '?', '[', '!', ']', 'x', '(', ')', 'p', 't',
                                   'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']

        names = []
        features = []
        labels = []
        for file in os.listdir(os.getcwd()+"/samples"):
            sig_name = file.split(".")[0]
            if sig_name not in names:
                names.append(sig_name)
                feature = fe.extract_multiclass_features(sig_name)
                label = fe.define_multiclass_labels(sig_name, multiclass_target_names)
                for sample in feature:
                    features.append(sample)
                for value in label:
                    labels.append(value)
        print(len(features) == len(labels))
        GridSearch(np.asarray(features), np.asarray(labels),['not QRS'] + multiclass_target_names)

