import wfdb
from scipy import signal
import numpy as np


class FeatureExtraction:

    def extract_features(self, sample_name):
        record = wfdb.rdsamp('samples/' + sample_name)
        annotation = wfdb.rdann('samples/' + sample_name, 'atr')
        peak_location = annotation.sample
        labels = []

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
            if i in peak_location:
                labels.append(1)
            else:
                labels.append(-1)
        np.asarray(features)
        np.asarray(labels)
        return features, labels
