import wfdb
from scipy import signal
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
import pickle

class FeatureExtraction:

    def extract_features(self,name, path, rpeak_locations, features_comb):
        if 'KNN_w' in features_comb:
            approach = 'window'
        else:
            approach = 'sample'
        if '1' in features_comb:
            channels = [0]
        elif '2' in features_comb:
            channels = [1]
        else:
            channels = [0, 1]
        filtered = 'FS' in features_comb
        write = not (filtered and approach == 'window' and not '2' in features_comb)
        print("Extracting features for signal " + path + "...")
        record = self.preprocess(channels, name, path)
        features_path = 'rpeakdetection/KNN/features/'
        if filtered:
            filtered_record = list()
            for i in range(len(channels)):
                filtered_channel = self.filter(record[i])
                filtered_record.append(filtered_channel)
            record = filtered_record
        comb_name = features_comb[0] + '_' + features_comb[1] + '_' + features_comb[2]
        feat_name = features_path + name + '_' + comb_name + '.pkl'
        labels_name = features_path + name + '_' + comb_name + '_labels.pkl'
        if write:
            if approach == 'window':
                features, labels = self.compute_sliding_features(record, rpeak_locations, window_size=100)
            else:
                features, labels = self.compute_sample_features(record, rpeak_locations)
            with open(feat_name, 'wb') as fid:
                pickle.dump(features, fid)
            with open(labels_name, 'wb') as fid:
                pickle.dump(labels, fid)
        else:
            features = pickle.load(
                open("rpeakdetection/KNN/features/"+name+"_100_"+str(len(channels))+'.pkl', 'rb'))
            labels = pickle.load(
                open("rpeakdetection/KNN/features/"+name+"_100_labels.pkl", 'rb'))
        return record, features, labels

    def preprocess(self, channels, name, path):
        if name == '114':
            if channels == [[0]]:
                record = wfdb.rdrecord(path, channels=[1])
            elif channels == [[1]]:
                record = wfdb.rdrecord(path, channels=[0])
            else:
                record = wfdb.rdrecord(path, channels=channels)
        else:
            record = wfdb.rdrecord(path, channels=channels)
        record = np.transpose(record.p_signal)
        return record

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

    def compute_sample_features(self, record, rpeak_locations):
        features = list()
        labels = list()
        i = 0
        while i< len(record[0]):
            feature = list()
            for id in range(len(record)):
                feature.append(record[id][i])
            features.append(feature)
            labels.append(int(i in rpeak_locations))
            i += 1
        return features, labels

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
        filtered = signal.filtfilt(a, b, record)
        vector = [1, 2, 0, -2, -1]
        int_c = 160 / fs
        # 5.1 since in signal 100 we must include 5
        b = interp1d(range(1, 6), [i * fs / 8 for i in vector])(np.arange(1, 5.1, int_c))
        ecg_d = signal.filtfilt(b, 1, filtered)
        # print(ecg_d[:5])
        ecg_d = ecg_d / np.max(ecg_d)
        ''' Squaring nonlinearly enhance the dominant peaks '''
        ecg_s = ecg_d ** 2
        return ecg_s


