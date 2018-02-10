from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch
import numpy as np

fe = FeatureExtraction()



class TwoClassClassification:
    def __init__(self):
        target_names = ["not QRS", "QRS"]
        features = fe.extract_features('100')
        twoclass_labels = fe.define_2class_labels('100')
        GridSearch(np.asarray(features), twoclass_labels, target_names)
