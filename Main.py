from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch

fe = FeatureExtraction()

features = fe.extract_features('100')
twoclass_labels = fe.define_2class_labels('100')
multiclass_labels = fe.define_multiclass_labels('100')
gs2 = GridSearch(features, multiclass_labels)
gs1 = GridSearch(features, twoclass_labels)

