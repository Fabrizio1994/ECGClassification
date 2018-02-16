import wfdb
from scipy import signal
import numpy as np


class FeatureExtraction:
    def extract_features(self, sample_name):
        print("Extracting features for signal"+sample_name+"...")
        record = wfdb.rdrecord('samples/' + sample_name)
        first_channel = []
        second_channel = []
        for elem in record.p_signal:
            first_channel.append(elem[0])
            second_channel.append((elem[1]))
        first_filtered = self.passband_filter(first_channel)
        second_filtered = self.passband_filter(second_channel)
        gradient_channel1 = np.gradient(first_filtered)
        gradient_channel2 = np.gradient(second_filtered)
        features = []
        for i in range(record.sig_len):
            print(i)
            features.append([gradient_channel1[i], gradient_channel2[i]])
        return features

    def passband_filter(self,record):
        lowpass = signal.butter(1, 12 / (360 / 2.0), 'low')
        highpass = signal.butter(1, 5 / (360 / 2.0), 'high')
        ecg_low = signal.filtfilt(*lowpass, x=record)
        return signal.filtfilt(*highpass, x=ecg_low)


    def extract_multiclass_features(self, sample_name ):
        print("Extracting features for signal" + sample_name + "...")
        record = wfdb.rdrecord('samples/' + sample_name)
        annotations = wfdb.rdann('samples/' + sample_name, 'atr')
        print(len(annotations.sample))
        first_channel = []
        second_channel = []
        samples = annotations.sample
        record.p_signal = signal.lfilter([-16, -32], [-1], record.p_signal)
        i = 0
        for elem in record.p_signal:
            if i in annotations.sample:
                first_channel.append(elem[0])
                second_channel.append((elem[1]))
            i += 1
        gradient_channel1 = np.gradient(first_channel)
        gradient_channel2 = np.gradient(second_channel)
        features = []
        for i in range(len(gradient_channel1)):
            features.append([gradient_channel1[i], gradient_channel2[i]])
        return features

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
        siglen = 650000
        annotation = wfdb.rdann('samples/' + sample_name, 'atr')
        labels = []
        symbols2id = {}

        for id in range(len(symbols)):
            symbols2id[symbols[id]] = id
        signal_symbols = annotation.symbol
        signal_samples = annotation.sample

        j = 0
        for i in range(siglen):
            if i in signal_samples:
                labels.append(symbols2id[signal_symbols[j]])
                j += 1
        return labels
