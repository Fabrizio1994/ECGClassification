import wfdb
from scipy import signal
import numpy as np


class FeatureExtraction:
    def extract_features(self, sample_name):
        print("Extracting features...")
        record = wfdb.rdsamp('samples/' + sample_name)
        first_channel = []
        second_channel = []
        record.p_signals = signal.lfilter([-16, -32], [-1], record.p_signals)
        for elem in record.p_signals:
            first_channel.append(elem[0])
            second_channel.append((elem[1]))
        gradient_channel1 = np.gradient(first_channel)
        gradient_channel2 = np.gradient(second_channel)

        features = []
        for i in range(650000):
            print(i)
            features.append([gradient_channel1[i], gradient_channel2[i]])
        return np.asarray(features)

    def define_2class_labels(self, sample_name):
        annotation = wfdb.rdann('samples/' + sample_name, 'atr')
        labels = []
        peak_location = annotation.sample
        for i in range(650000):
            if i in peak_location:
                labels.append(1)
            else:
                labels.append(-1)
        return np.asarray(labels)

    def define_multiclass_labels(self, sample_name, symbols):
        annotation = wfdb.rdann('samples/' + sample_name, 'atr')
        labels = []
        symbols2id = {}

        for id in range(len(symbols)):
            symbols2id[symbols[id]] = id

        signal_symbols = annotation.symbol
        signal_samples = annotation.sample
        len(signal_symbols)

        j = 0
        for i in range(650000):
            if i in signal_samples:
                labels.append(signal_symbols[j])
                j += 1
            else:
                labels.append('$')
        return np.asarray(labels)
