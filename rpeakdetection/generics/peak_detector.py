import wfdb
import peakutils
from rpeakdetection.Utility import Utility
import numpy as np
from rpeakdetection.pan_tompkins.rpeak_detector import RPeakDetector
import matplotlib.pyplot as plt

PATH = 'data/ecg/mitdb/'
records = wfdb.get_record_list('mitdb')
util = Utility()
rpeak = RPeakDetector()
eval_width = 36

prec_means = []
rec_means = []

for thresh in np.arange(0.1, 1.0, 0.05):
    recalls = []
    precisions = []
    for name in records:
        record = wfdb.rdrecord(PATH + name, channels=[0])
        peaks_locations = util.remove_non_beat(PATH + name, rule_based=True)[0]
        record = record.p_signal.flatten()
        record = np.abs(record)
        indices = peakutils.indexes(record, thres=thresh, min_dist=40)
        recall, precision = rpeak.evaluate(indices, PATH + name, eval_width, rule_based=True)
        precisions.append(precision)
        recalls.append(recall)
    print('THRESHOLD IS: ' + str(thresh))
    print('PRECISION IS: ' + str(np.mean(precisions)))
    prec_means.append(np.mean(precisions))
    print('RECALL IS: ' + str(np.mean(recalls)))
    rec_means.append(np.mean(recalls))

plt.plot(np.arange(0.1, 1.0, 0.05), prec_means)
plt.plot(np.arange(0.1, 1.0, 0.05), rec_means)
plt.show()

plt.savefig('prec-rec-threshold-generic.png')
