import wfdb
import numpy as np
from scipy import stats
import os
from beatclassification.LabelsExtraction import LabelsExtraction
from sklearn.decomposition import PCA

peaks_path = "../data/peaks/pantompkins/mitdb"
ann_path = "../../data/ecg/mitdb/"
LEFT_WINDOW = 70
RIGHT_WINDOW = 100
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

    def extract(self, names, scale_factors=None, one_hot=False, read_annot=False):
        # extract labels of peaks from rpeakdetector
        if not read_annot:
            # dict -> signame2symbols
            self.symbols = le.extract(peaks_path)
        features = list()
        labels = list()
        for name in names:
            print(name)
            signal, peaks, symbols = self.read_data(name, read_annot)
            rr_intervals = np.diff(peaks)
            rr_mean = np.mean(rr_intervals)
            # symbols out of the scope of this study are excluded
            symbols = list(filter(lambda x: x in self.mitdb_symbols, symbols))
            # because the 5 previous rr intervals are needed, the first rr interval is also excluded
            # because its value is not relevant(it is the distance from the start).
            for count in range(6, len(symbols)-5):
                symbol = symbols[count]
                feature = list()
                peak = peaks[count]
                qrs = signal[peak - LEFT_WINDOW:peak +RIGHT_WINDOW]
                feature = self.rr_features(count, feature, rr_intervals, rr_mean)
                feature = self.signal_cumulants(qrs, feature)
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
        prev_3 = rr_intervals[count - 3: count]
        prev_3 = np.divide(prev_3, rr_mean)
        prev = rr_intervals[count - 1]
        next = rr_intervals[count]
        feature.extend([prev, next, win_mean])
        feature.extend(prev_3)
        return feature

    def read_data(self, name, read_annot):
        record = wfdb.rdrecord(ann_path + name)
        signal = []
        for elem in record.p_signal:
            signal.append(elem[0])
        if read_annot:
            annotation = wfdb.rdann(ann_path + name, "atr")
            symbols = annotation.symbol
            peaks = annotation.sample
        else:
            symbols = self.symbols[name]
            peaks = self.read_peaks(peaks_path, name)
        return signal, peaks, symbols

    # takes a window of [-90,+90] around the Rpeak
    # TODO: does it require filtering?
    def signal_cumulants(self, qrs, feature):
        poses = [30, 60, 90, 120, 150]
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


    def read_peaks(self, peaks_path, name):
        file = open(peaks_path + "/" + name + ".tsv")
        peaks = []
        for line in file:
            peaks.append(int(line.replace("\n","")))
        return peaks

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




