import wfdb
from scipy import signal
import numpy as np
from collections import defaultdict
import pickle

class FeatureExtraction:

    def extract_features(self,name, path, rpeak_locations, window_size, write=False):
        print("Extracting features for signal " + path + "...")
        if name == '114':
            record = wfdb.rdrecord(path, channels=[1])
        else:
            record = wfdb.rdrecord(path, channels=[0])
        record = np.transpose(record.p_signal)
        if write:
            gradient = self.gradient(self.filter(record))
            features, labels = self.compute_sliding_features(gradient, rpeak_locations, window_size)
            with open('rpeakdetection/KNN/features/' + name +'.pkl', 'wb') as fid:
                pickle.dump(features, fid)
            with open('rpeakdetection/KNN/features/' + name + '_labels.pkl', 'wb') as fid:
                pickle.dump(labels, fid)
        else:
            features = pickle.load(
                open('rpeakdetection/KNN/features/'+ name +'.pkl', 'rb'))
            labels = pickle.load(
                open('rpeakdetection/KNN/features/' + name + '_labels.pkl', 'rb'))
        return record, features, labels

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
        return features, labels

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


