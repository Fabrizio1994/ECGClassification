import wfdb
from scipy import signal
import numpy as np
from collections import defaultdict

class FeatureExtraction:

    def extract_features(self, signal_name, rpeak_locations, window_size, feature_group):
        print("Extracting features for signal " + signal_name + "...")
        record = wfdb.rdrecord(signal_name, channels=[0])
        record = np.transpose(record.p_signal)
        record = self.gradient(self.filter(record))
        if feature_group == 'window':
            return self.compute_sliding_features(record, rpeak_locations, window_size)


    def compute_sliding_features(self, record, rpeak_locations, window_size):
        features = list()
        labels = list()
        i = 0
        while i <len(record[0]) - window_size:
            feature = list()
            for id in range(len(record)):
                feature.extend([record[id][j] for j in range(i, i+ window_size)])
            # checks whether the current window contains an annotation
            annotated = True in list(map(lambda x: x in rpeak_locations, range(i, i+ window_size)))
            features.append(feature)
            if annotated:
                labels.append(1)
            else:
                labels.append(-1)
            i += window_size
        return record, features, labels

    def gradient(self, record):
        record_gradient = list()
        for i in range(len(record)):
            gradient = np.diff(record[i])
            gradient_norm = np.max(np.abs(gradient))
            record_gradient.append(np.divide(gradient, gradient_norm))
        return record_gradient


    def filter(self, record):
        fs = 360
        # cutoff low frequency to get rid of baseline wonder
        f1 = 5
        # cutoff frequency to discard high frequency noise
        f2 = 15
        Wn = np.divide([f1*2, f2*2],fs)
        N = 3
        # bandpass
        [a, b] = signal.butter(N, Wn, 'band')
        # filtering
        for i in range(len(record)):
            record[i] = signal.filtfilt(a, b, record[i])
        return record


