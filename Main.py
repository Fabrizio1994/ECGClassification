from FeatureExtraction import FeatureExtraction
from RPeakEvaluation import RPeakEvaluation
from GridSearch import GridSearch
import os

fe = FeatureExtraction()
rp = RPeakEvaluation()

rp.validate_r_peak()


#for name in os.listdir("features"):
#    signal_name = name.replace(".tsv","")
#    train_features, train_labels = fe.extract_features("sample/"+signal_name)
#    GridSearch(signal_name, train_features, train_labels, ["not QRS", "QRS"])


