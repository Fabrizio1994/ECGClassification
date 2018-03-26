import wfdb
from scipy import signal
import numpy as np
from collections import defaultdict

class FeatureExtraction:
    def extract_features(self, sample_name, annotation_type, window_size,
                         features_type="fixed", channels_ids=[0, 1]):
        print("Extracting features for signal " + sample_name + "...")
        record = wfdb.rdrecord(sample_name)
        annotation = wfdb.rdann('annotations/' + annotation_type + '/' +
                                sample_name.replace("sample/", ""), 'atr')

        channels_map = defaultdict(list)
        for elem in record.p_signal:
            for id in channels_ids:
                channels_map[id].append(elem[id])

        for channel in channels_map:
            channels_map[channel] = self.__normalized_gradient(self.__passband_filter(channels_map[channel]))

        print("actual peaks:" + str(len(annotation.sample)))
        if features_type == "sliding":
            return self.__compute_sliding_features(channels_map, annotation, window_size)
        return self.__compute_fixed_features(channels_map, annotation, window_size)

    def __compute_fixed_features(self, channels_map, annotation, window_size):
        features = []
        siglen = len(channels_map[0])
        for i in range(siglen):
            features.append([channels_map[channel_id][i] for channel_id in channels_map.keys()])
        samples = annotation.sample
        labels = [-1] * siglen
        for j in range(len(samples)):
            annotated_index = j
            qrs_region = self.__get_qrs_region(samples, annotated_index, window_size, siglen)
            for sample in qrs_region:
                labels[sample] = 1
        return np.asarray(features), np.asarray(labels)

    def __compute_sliding_features(self, channels_map, annotation, window_size):
        samples = annotation.sample
        features = []
        labels = []
        i = 0
        while i < len(channels_map[0]) - window_size:
            feature = []
            annotated = False
            for j in range(i, i + window_size):
                if j in samples:
                    annotated = True
                for id in channels_map.keys():
                    feature.append(channels_map[id][j])
            features.append(feature)
            if annotated:
                labels.append(1)
            else:
                labels.append(-1)
            i += window_size
        return np.asarray(features), np.asarray(labels)

    def __get_qrs_region(self, samples, annotated_index, window_size, siglen):
        boundary = int(window_size/2)
        if samples[annotated_index] < siglen - boundary:
            if samples[annotated_index] - boundary > 0:
                return [q for q in range(samples[annotated_index] - boundary +1 ,
                                         samples[annotated_index] + boundary + 1)]
            else:
                return [q for q in range(samples[annotated_index] + boundary + 1)]
        else:
            gap = siglen - samples[annotated_index]
            return [q for q in range(samples[annotated_index] - boundary,
                                     samples[annotated_index] + gap)]



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


