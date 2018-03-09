
from Utility import Utility
#ut = Utility()

#ut.run_knn()

from FeatureExtraction import FeatureExtraction
fe = FeatureExtraction()
fe.extract_features("sample/100","cleaned", 50, features_type="on_annotation")


