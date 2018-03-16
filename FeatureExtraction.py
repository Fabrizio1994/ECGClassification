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
        filtered_first_channel = self.__passband_filter(first_channel)
        filtered_second_channel = self.__passband_filter(second_channel)
        gradient_channel1 = self.__normalized_gradient(filtered_first_channel)
        gradient_channel2 = self.__normalized_gradient(filtered_second_channel)
        print("actual peaks:" + str(len(annotation.sample)))
        if features_type == 'on_annotation':
            return self.__compute_on_annotation_features(gradient_channel1, gradient_channel2, annotation)
        elif features_type == "sliding":
            return self.__compute_sliding_features(gradient_channel1,
                                                   gradient_channel2,
                                                   annotation,
                                                   window_size)
        return self.__compute_fixed_features(gradient_channel1, gradient_channel2, annotation, window_size)

    def __compute_fixed_features(self, channel1, channel2, annotation, window_size):
        features = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
        samples = annotation.sample
        labels = [-1] * len(channel1)
        for j in range(len(samples)):
            siglen = len(channel1)
            annotated_index = j
            qrs_region = self.__get_qrs_region(samples, annotated_index, window_size, siglen)
            for sample in qrs_region:
                labels[sample] = 1
        return np.asarray(features), np.asarray(labels)

    def __compute_on_annotation_features(self, channel1, channel2, annotation):
        features = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
        samples = annotation.sample
        labels = [-1] * len(channel1)
        for j in range(len(samples)):
            labels[samples[j]] = -1
        return np.asarray(features), np.asarray(labels)

    def __compute_sliding_features(self, channel1, channel2, annotation,
                                   window_size):
        samples = annotation.sample
        features = []
        labels = []
        i = 0
        while i < len(channel1) - window_size:
            feature = []
            annotated = False
            for j in range(i, i +window_size):
                if j in samples:
                    annotated = True
                feature.append(channel1[j])
                feature.append(channel2[j])
            features.append(feature)
            if annotated:
                labels.append(1)
            else:
                labels.append(-1)
            i += window_size
        return np.asarray(features), np.asarray(labels)

    def __get_qrs_region(self, samples, annotated_index, window_size, siglen):
        boundary = int(window_size/2)
        if samples[annotated_index] <= siglen - boundary:
            if samples[annotated_index] - boundary > 0:
                return [q for q in range(samples[annotated_index] - boundary,
                                         samples[annotated_index] + boundary + 1)]
            else:
                return [q for q in range(samples[annotated_index] + boundary + 1)]
        else:
            gap = siglen - samples[annotated_index]
            return [q for q in range(samples[annotated_index] - boundary,
                                     samples[annotated_index] + gap + 1)]



    def __normalized_gradient(self, channel):
        gradient = np.diff(channel)
        gradient_norm = np.sqrt(np.sum(np.square(gradient)))
        normalized_gradient = np.divide(gradient, gradient_norm)
        return normalized_gradient


    def __passband_filter(self, channel):
        freq = 360.0 / 2.0
        b, a = signal.butter(1, [5 / freq, 12 / freq], btype="bandpass")
        new_channel = signal.filtfilt(b, a, channel)
        return new_channel


