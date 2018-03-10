import wfdb
from scipy import signal
import numpy as np


class FeatureExtraction:
    def extract_features(self, sample_name, annotation_type, window_size,
                         features_type="fixed"):
        print("Extracting features for signal " + sample_name + "...")
        record = wfdb.rdrecord(sample_name)
        annotation = wfdb.rdann('annotations/' + annotation_type + '/' +
                                sample_name.replace("sample/", ""), 'atr')
        first_channel = []
        second_channel = []
        for elem in record.p_signal:
            first_channel.append(elem[0])
            second_channel.append(elem[1])
        filtered_first_channel = self.passband_filter(first_channel)
        filtered_second_channel = self.passband_filter(second_channel)
        gradient_channel1 = self.normalized_gradient(filtered_first_channel)
        gradient_channel2 = self.normalized_gradient(filtered_second_channel)
        print("actual peaks:" + str(len(annotation.sample)))
        if features_type == 'on_annotation':
            return self.compute_on_annotation_features(gradient_channel1,
                                                       gradient_channel2,
                                                       annotation)
        return self.compute_fixed_features(gradient_channel1, gradient_channel2,
                                           annotation, window_size)

    def compute_fixed_features(self, channel1, channel2, annotation, window_size):
        features = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
        samples = annotation.sample
        labels = [-1] * len(channel1)
        for j in range(len(samples)):
            siglen = len(channel1)
            annotated_index = j
            qrs_region = self.get_qrs_region(samples, annotated_index, window_size, siglen)
            for sample in qrs_region:
                labels[sample] = 1
        return np.asarray(features), np.asarray(labels)

    def compute_on_annotation_features(self, channel1, channel2, annotation):
        features = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
        samples = annotation.sample
        labels = [-1] * len(channel1)
        for j in range(len(samples)):
            labels[samples[j]] = -1
        return np.asarray(features), np.asarray(labels)

    def get_qrs_region(self, samples, annotated_index, window_size, siglen):
        if annotated_index < siglen - (int(window_size / 2) + 1):
            return [q for q in range(samples[annotated_index] - (int(window_size / 2)),
                                     samples[annotated_index] + int(window_size / 2) + 1)]

        else:
            gap = siglen - annotated_index - 1
            return [q for q in range(samples[annotated_index] - gap,
                                     samples[annotated_index] + gap)]

    def normalized_gradient(self, channel):
        gradient = np.diff(channel)
        gradient_norm = np.sqrt(np.sum(np.square(gradient)))
        normalized_gradient = np.divide(gradient, gradient_norm)
        return normalized_gradient


    def passband_filter(self, channel):
        freq = 360.0 / 2.0
        b, a = signal.butter(1, [5 / freq, 12 / freq], btype="band")
        new_channel = signal.filtfilt(b, a, channel)
        return new_channel


