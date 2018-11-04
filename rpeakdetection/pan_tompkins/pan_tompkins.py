import os
import numpy as np
from rpeakdetection.pan_tompkins.rpeak_detector import RPeakDetector
import wfdb
rpd = RPeakDetector()
evaluation_width = 36
ecg_folder = "data/ecg/mitdb/"
peaks_folder = "data/peaks/pantompkins/mitdb/"
precisions = list()
recalls = list()
for name in wfdb.get_record_list('mitdb'):
    peaks = list()
    file = open(peaks_folder + name + '.tsv', "r")
    print(name)
    for line in file:
        peak = line.replace("\n", "")
        peaks.append(int(peak))
    recall, precision = rpd.evaluate(peaks, ecg_folder + name, evaluation_width, rule_based=True)

    print('recall : ' + str(recall))
    print('precision : ' + str(precision))


    precisions.append(precision)
    recalls.append(recall)
print("av prec")
print(np.mean(precisions))
print("av recall")
print(np.mean(recalls))

