from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch

fe = FeatureExtraction()

features, labels = fe.extract_features('100')
gs = GridSearch(features, labels)

