import wfdb
import peakutils
from rpeakdetection.Utility import Utility
import numpy as np
from rpeakdetection.rpeak_detector import RPeakDetector
import matplotlib.pyplot as plt
from collections import defaultdict

PATH = 'data/ecg/mitdb/'
util = Utility()
rpeak = RPeakDetector()
eval_width = 36
class PeakDetector():

    def choose_tresholds(self, thresholds):
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for name in wfdb.get_record_list('mitdb'):
            print(name)
            for thresh in thresholds:
                record, indices = self.detect_peaks(name, thresh)
                recall, precision = rpeak.evaluate(indices, PATH + name, eval_width, rule_based=False)
                precisions[thresh].append(precision)
                recalls[thresh].append(recall)
        average_prec = [np.mean(precisions[t]) for t in thresholds]
        average_rec = [np.mean(recalls[t]) for t in thresholds]
        thresh_index = np.argmax([(average_prec[j] + average_rec[j])/2 for j in range(len(average_rec))])
        best_threshold = thresholds[thresh_index]
        plt.plot([best_threshold]*2, [0, 1], label='best_threshold')
        plt.plot(thresholds, average_prec, label = 'precision')
        plt.plot(thresholds, average_rec, label = 'recall')
        plt.xlabel('threshold')
        plt.ylabel('precision/recall')
        plt.legend()
        print(average_rec)
        print(average_prec)
        plt.savefig('prec-rec-threshold-generic.png')
        return best_threshold

    def detect_peaks(self, name, thresh):
        record = self.preprocess(name)
        return record, peakutils.indexes(record, thres=thresh, min_dist=40)

    def preprocess(self, name):
        record = wfdb.rdrecord(PATH + name, channels=[0])
        record = record.p_signal.flatten()
        record = np.abs(record)
        record = np.divide(record, np.max(record))
        return record

    def plot_criticism(self, peaks, signal, plot_to, threshold,  plot_from=0):
        real_peaks = list(filter(lambda x : x <= peaks[-1] + 18,util.remove_non_beat('data/ecg/mitdb/100', False)[0]))
        fig, ax = plt.subplots()
        ax.scatter(real_peaks, [signal[p] for p in real_peaks], color ='blue', label = 'real_peaks')
        ax.scatter([peaks], [signal[peaks]], color ='red', label = 'detected_peak')
        plt.plot(signal[plot_from:plot_to])
        plt.plot([0, plot_to], [threshold]*2, label = 'threshold')
        ax.legend()
        plt.legend()
        plt.savefig("generic_criticism.png")
        plt.close()

    def signals_evaluation(self, threshold):
        precisions = list()
        recalls = list()
        for name in wfdb.get_record_list('mitdb'):
            record, indices = self.detect_peaks(name, threshold)
            precision, recall = rpeak.evaluate(indices, PATH +name, eval_width, False)
            precisions.append(precision)
            recalls.append(recall)
            print('{:s}, {:f}, {:f}'.format(name, precision, recall))
        av = 'average'
        print("{:s}, {:f}, {:f}".format(av, np.mean(precisions), np.mean(recalls)))


if __name__ == '__main__':
    pd = PeakDetector()
    pd.choose_tresholds(np.arange(0.1,0.95,0.05))