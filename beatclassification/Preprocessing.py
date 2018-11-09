import wfdb
from rpeakdetection.Utility import Utility
import numpy as np
from collections import defaultdict
import random
from scipy import signal
from scipy.signal import medfilt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

ut = Utility()
ecg_path = 'data/ecg/mitdb/'
# classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F"}
# '/': 'Q', 'f': 'Q', 'Q': 'Q'}

sig_len = 650000


class Preprocessing():

    def preprocess(self, dataset_names, X_shape, filtered=False, classes=None, aami=True, one_hot=True, timesteps=None):
        if classes is None:
            classes = ['N', 'S', 'V', 'F']
        X = np.empty(X_shape)
        if one_hot:
            Y = np.empty((X_shape[0], len(classes)))
        else:
            Y = np.empty(X_shape[0], dtype='int8')
        count = 0
        for name in dataset_names:
            # print(name)
            if name != '114':
                record = wfdb.rdrecord(ecg_path + name, channels=[0])
            else:
                record = wfdb.rdrecord(ecg_path + name, channels=[1])
            peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
            # noinspection PyArgumentList
            record = np.transpose(record.p_signal)
            record = record[0]
            record = self.remove_baseline(record)
            peaks, symbols = self.exclude_beats(peaks, symbols)
            labels = self.extract_labels(aami, classes, one_hot, symbols)
            beats = [record[p - 70:p + 100] for p in peaks]
            X[count:count + len(peaks)] = beats
            Y[count:count + len(peaks)] = labels
            count += len(peaks)
        if timesteps is not None:
            X, Y  = self.compute_timesteps(X,Y, timesteps)
        return X, Y

    def compute_timesteps(self, X, Y, timesteps):
        end_beats = timesteps -1
        start_label = end_beats
        windowed = np.zeros((X.shape[0] - end_beats, timesteps, X.shape[1]))
        for i in range(len(X) - end_beats):
            window = [X[w] for w in range(i, i + timesteps)]
            windowed[i] = window
        labels = Y[start_label:]
        return windowed, labels

    def exclude_beats(self, peaks, symbols):
        # exclude beats at the end of the signal (padding?), 21 in total
        pairs = list(filter(lambda x: 70 <= x[0] < sig_len - 100, zip(peaks, symbols)))
        peaks, symbols = zip(*pairs)
        return peaks, symbols

    def extract_labels(self, aami, classes, one_hot, symbols):
        symbols = list(symbols)
        labels = np.zeros(len(symbols), dtype='int8')
        for i in range(len(symbols)):
            if symbols[i] in symbol2class.keys():
                if aami:
                    classe = symbol2class[symbols[i]]
                    labels[i] = classes.index(classe)
                else:
                    labels[i] = classes.index(symbols[i])
            else:
                labels[i] = 0
        if one_hot:
            one_hot_sym = np.zeros((len(symbols), len(classes)))
            for one_hot, sym in zip(one_hot_sym, labels):
                one_hot[sym] = 1
            labels = one_hot_sym
        return labels

    def subsample_data(self, X, Y, classes, label, factor, one_hot):
        if one_hot:
            label_data = list(filter(lambda x: np.argmax(x[1]) == classes.index(label), zip(X, Y)))
            other_data = list(filter(lambda x: np.argmax(x[1]) != classes.index(label), zip(X, Y)))
        else:
            label_data = list(filter(lambda x: x[1] == classes.index(label), zip(X, Y)))
            other_data = list(filter(lambda x: x[1] != classes.index(label), zip(X, Y)))
        label_X, label_Y = zip(*label_data)
        X, Y = zip(*other_data)
        sample_len = int(len(label_X) / factor)
        label_X = random.sample(label_X, sample_len)
        label_Y = label_Y[:sample_len]
        X = np.concatenate((X, label_X))
        Y = np.concatenate((Y, label_Y))
        return X, Y

    def augment_data(self, X, Y, classes, label, factor, one_hot=True):
        if one_hot:
            label_data = list(filter(lambda x: np.argmax(x[1]) == classes.index(label), zip(X, Y)))
        else:
            classe = classes.index(label)
            print(classe)
            label_data = list(filter(lambda x: x[1] == classe, zip(X, Y)))
        label_X, label_Y = zip(*label_data)
        for i in range(factor):
            X = np.concatenate((X, list(map(lambda x: x + random.random(), label_X))))
            Y = np.concatenate((Y, label_Y))
        return X, Y

    def filter(self, channel):
        fs = 360
        # cutoff low frequency to get rid of baseline wonder
        f1 = 5
        # cutoff frequency to discard high frequency noise
        f2 = 15
        Wn = np.divide([f1 * 2, f2 * 2], fs)
        N = 3
        # bandpass
        [a, b] = signal.butter(N, Wn, 'band')
        # filtering
        filtered = signal.filtfilt(a, b, channel)
        return filtered

    def read_image(self, dataset_names, X_shape, classes=None, one_hot=True, aami=True):
        if classes is None:
            classes = ['N', 'S', 'V', 'F']
        X = np.empty(X_shape)
        if one_hot:
            Y = np.empty((X_shape[0], len(classes)))
        else:
            Y = np.empty(X_shape[0], dtype='int8')
        count = 0
        for name in dataset_names:
            print(name)
            record = wfdb.rdrecord(ecg_path + name, channels=[0, 1])
            record = np.transpose(record.p_signal)
            # for i in range(len(record)):
                # record[i] = self.remove_baseline(record[i])
            peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
            peaks, symbols = self.exclude_beats(peaks, symbols)
            labels = self.extract_labels(aami=aami, classes=classes, one_hot=one_hot, symbols=symbols)
            beats = np.zeros((len(peaks), X_shape[1], X_shape[2], X_shape[3]))
            for index, p in enumerate(peaks):
                beat = np.zeros((X_shape[1], X_shape[2], X_shape[3]))
                for i in range(len(record)):
                    channel = record[i]
                    beat_data = channel[p - 70:p + 100]
                    beat[i] = np.reshape(beat_data, (beat.shape[1], beat.shape[2]))
                beats[index] = beat
            X[count: count + len(peaks)] = beats
            Y[count: count + len(peaks)] = labels
            count += len(peaks)
        return X, Y

    def segment(self, classes):
        X = list()
        Y = list()
        for name in ['232']:
            if name != '114':
                channels = [0]
            else:
                channels = [1]
            record = wfdb.rdrecord(ecg_path + name, channels=channels)
            record = np.transpose(record.p_signal)
            record = record[0]
            record = self.remove_baseline(record)
            peaks, symbols = ut.remove_non_beat(ecg_path + name, rule_based=False, include_VF=True)
            data = list(filter(lambda x: x[1] in classes, zip(peaks, symbols)))
            peaks, symbols = zip(*data)
            for i, p in enumerate(peaks[1:-1], start=1):
                left_end = peaks[i - 1] + 20
                right_end = peaks[i + 1] - 20
                beat = record[left_end:right_end]
                label = symbols[i]
                index = classes.index(label)
                X.append(beat)
                Y.append(index)
        print(len(X))
        print(len(Y))
        print(set(Y))
        return X, Y

    def remove_baseline(self, record):
        # 200 ms -> 71 samples with fs=360
        baseline = medfilt(record, 71)
        # 600 ms -> 215 samples with fs=360
        baseline = medfilt(baseline, 215)
        # remove baseline wander
        record = np.subtract(record, baseline)
        return record

    def standardize(self, X):
        if len(X.shape) == 2:
            scalers = StandardScaler()
            scalers.fit(X)
            scaled = scalers.transform(X)
            return scaled, scalers
        else:
            mean = np.mean(X, axis =(1,2), keepdims=True)
            std = np.std(X, axis=(1,2), keepdims=True)
            standardized = (X -mean)/std
            return standardized
