from FeatureExtraction import FeatureExtraction
from GridSearch import GridSearch

fe = FeatureExtraction()

twoclass_target_names = ["QRS", "not QRS"]
multiclass_target_names = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j'
                    , 'n', 'E', '/', 'f', 'Q', '?','[', '!', ']', 'x', '(', ')', 'p', 't',
                    'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D','=', '"', '@']
features = fe.extract_features('100')
twoclass_labels = fe.define_2class_labels('100')
multiclass_labels = fe.define_multiclass_labels('100', multiclass_target_names)

gs1 = GridSearch(features, twoclass_labels, twoclass_target_names)
gs2 = GridSearch(features, multiclass_labels, multiclass_target_names)

