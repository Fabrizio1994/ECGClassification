import os
import numpy as np
from rpeakdetection.rpeak_detector import RPeakDetector
rpd = RPeakDetector()
evaluation_width = 36
ecg_folder = "../data/ecg/mitdb/"
peaks_folder = "../data/peaks/pan_tompkins/"
precisions = list()
recalls = list()
for name in os.listdir(peaks_folder):
    peaks = list()
    file = open(peaks_folder + name, "r")
    name = name.replace(".tsv", "")
    for line in file:
        peak = line.replace("\n", "")
        peaks.append(int(peak))
    precision, recall = rpd.evaluate(peaks, ecg_folder + name, evaluation_width )
    precisions.append(precision)
    recalls.append(recall)
print("av prec")
print(np.mean(precisions))
print("av recall")
print(np.mean(recalls))