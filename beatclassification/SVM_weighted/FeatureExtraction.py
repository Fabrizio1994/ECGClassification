import wfdb
import numpy as np
from scipy import stats
from scipy.signal import medfilt

import os
from beatclassification.LabelsExtraction import LabelsExtraction
import pywt


le = LabelsExtraction()

class FeatureExtraction:
    def __init__(self):
        # classes of beats
        self.classes = ["N", "S", "V", "F"]
        # physiobank symbols for beats
        self.mitdb_symbols = ["N", "L", "R", "e", "J", "A", "a", "J", "S", "V", "E", "F"]
        # associates symbols to classes of beats
        self.symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                             "A": "S", "a": "S", "J": "S", "S": "S",
                             "V": "V", "E": "V",
                             "F": "F"}
        # label is the integer value associated to a class
        self.class2label = {}
        self.label2class = {}

        count = 0
        for classe in self.classes:
            self.class2label[classe] = count
            self.label2class[count] = classe
            count += 1

    def extract(self, db_names, peaks, ann_path, features_group=['rr'], from_annot=True, left_window=70, right_window=100,
                scale_factors=None, one_hot=False):
        """
        :param features_group: list or numpy array containing the feature names to compute.
            available = 'rr', 'hos', 'wavelets', 'raw'
        :param db_names: list or numpy array containing the names of the signals in the train or test database
        :param peaks: dict[signal_name, locations] containing the locations of the peaks in the signal
        :param ann_path: path of the local folder containing the annotations files (.atr)
        :param from_annot: whether to associate labels to peaks from the ground truth(.atr file)
        :param left_window: number of samples at the left of the rpeak location to include in the features computation
        :param right_window: number of samples at the right of the rpeak location to include in the features computation
        :param scale_factors: size=n_classes
            list or numpy array containing the reducing or augmenting factors for each class
        :param one_hot: whether to return labels in one hot format
        :return: features and labels arrays for the given database(train or test)
        """
        features = list()
        labels = list()
        symbols = le.extract(ann_path, from_annot=from_annot)
        for name in db_names:
            print(name)
            sig_peaks = peaks[name]
            signal, sig_symbols = self.read_data(name, ann_path, symbols)
            rr_intervals = np.diff(sig_peaks)
            rr_mean = np.mean(rr_intervals)
            # starts from the 6th beat because the 5 previous rr intervals are needed
            # ends at 5 to the end beacause the 5 following rr intervals are needed
            sig_symbols = sig_symbols[5:-5]
            for count in range(5, len(sig_symbols)):
                symbol = sig_symbols[count]
                if symbol in self.mitdb_symbols:
                    feature = list()
                    peak = sig_peaks[count]
                    qrs = signal[peak - left_window:peak + right_window]
                    if 'raw' in features_group:
                        feature.extend(qrs)
                    if 'rr' in features_group:
                        feature = self.rr_features(count, feature, rr_intervals, rr_mean)
                    if 'hos' in features_group:
                        feature = self.signal_cumulants(qrs, feature)
                    if 'wavelets' in features_group:
                        feature = self.wavelets(qrs, feature)
                    features.append(feature)
                    classe = self.symbol2class[symbol]
                    label = self.class2label[classe]
                    if one_hot:
                        hot_lab = np.zeros(len(self.class2label), dtype=np.int8)
                        hot_lab[label] = 1
                        label = hot_lab
                    labels.append(label)
        if scale_factors is not None:
            features, labels = self.resample(features, labels, scale_factors=scale_factors)
        return np.array(features), np.array(labels)

    def rr_features(self, count, feature, rr_intervals, rr_mean):
        window = rr_intervals[count - 5: count + 5]
        win_mean = np.mean(window)
        prev_3 = rr_intervals[count - 2: count + 1]
        prev_3 = np.divide(prev_3, rr_mean)
        prev = rr_intervals[count]
        next = rr_intervals[count + 1]
        feature.extend([prev, next, win_mean])
        feature.extend(prev_3)
        return feature

    def read_data(self, name, ann_path, symbols):
        if name == '114':
            record = wfdb.rdrecord(ann_path + name, channels=[1])
        else:
            record = wfdb.rdrecord(ann_path + name, channels=[0])
        signal = record.p_signal.flatten()
        # median_filter1D
        baseline = medfilt(signal, 71)
        baseline = medfilt(baseline, 215)
        for i in range(0, len(signal)):
            signal[i] = signal[i] - baseline[i]
        return signal, symbols[name]

    # takes a window of [-90,+90] around the Rpeak
    # TODO: does it require filtering?
    def signal_cumulants(self, qrs, feature):
        poses = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
        lag = 15
        second = list()
        third = list()
        fourth = list()
        for pose in poses:
            window = qrs[pose - lag:pose + lag]
            cumulant_sample = np.var(window)
            second.append(cumulant_sample)
            cumulant_sample = stats.skew(window)
            third.append(cumulant_sample)
            cumulant_sample = stats.kurtosis(window, fisher=False)
            fourth.append(cumulant_sample)
        # normalization step
        second = np.divide(second, np.sqrt(sum(np.square(second))))
        third = np.divide(third, np.sqrt(sum(np.square(third))))
        fourth = np.divide(fourth, np.sqrt(sum(np.square(fourth))))
        feature.extend(second)
        feature.extend(third)
        feature.extend(fourth)
        return feature

    def resample(self, X_train, Y_train, scale_factors):
        count = 0
        X = list()
        Y = list()
        for j in range(len(X_train)):
            label = Y_train[j]
            # undersampling
            if scale_factors[label] < 0:
                if count == abs(scale_factors[label]):
                    X.append(X_train[j])
                    Y.append(Y_train[j])
                    count = 0
                else:
                    count += 1
            # oversampling
            else:
                X.extend([X_train[j]]*scale_factors[label])
                Y.extend([Y_train[j]]*scale_factors[label])
        return X, Y

    def wavelets(self, qrs, feature):
        db1 = pywt.Wavelet('db1')
        coeffs = pywt.wavedec(qrs, db1, level=3)
        wavel = coeffs[0]
        feature.extend(wavel)
        return feature




