from FeatureExtraction import FeatureExtraction
from RPeakEvaluation import RPeakEvaluation
from Utility import Utility
from GridSearch import GridSearch
import os

fe = FeatureExtraction()
rp = RPeakEvaluation()
ut = Utility()
#rp.validate_r_peak()
ut.clean_all()
#ut.clean_signal('sample/108')

#for name in os.listdir("features"):
#    signal_name = name.replace(".tsv","")
#    train_features, train_labels = fe.extract_features("sample/"+signal_name)
#    GridSearch(signal_name, train_features, train_labels, ["not QRS", "QRS"])


