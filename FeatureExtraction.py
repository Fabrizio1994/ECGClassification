import wfdb
from scipy import signal
import numpy as np
import os


class FeatureExtraction:
    def extract_features(self, sample_name, annotation_type, window_size,
                         features_type="fixed"):
        print("Extracting features for signal " + sample_name + "...")
        record = wfdb.rdrecord(sample_name)
        annotation = wfdb.rdann('annotations/' + annotation_type + '/' +
                                sample_name.replace("sample/", "") , 'atr')
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
        if features_type == "sliding":
            return self.compute_sliding_features(gradient_channel1, gradient_channel2,
                                                 annotation, window_size)
        return self.compute_features(gradient_channel1, gradient_channel2, annotation,
                                     window_size)

    def compute_sliding_features(self, channel1, channel2, annotation, window_size):
        features = []
        labels = []
        sample = annotation.sample
        j = 0
        while j < len(channel1) - window_size:
            feature = []
            qrs = False
            for i in range(j, j +window_size):
                feature.append(channel1[i])
                feature.append(channel2[i])
                if i in sample:
                    qrs = True
            features.append(feature)
            if qrs:
                labels.append(1)
            else:
                labels.append(-1)
            j += 1
        return np.asarray(features), np.asarray(labels)

    def compute_features(self, channel1, channel2, annotation, window_size):
        features = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
        samples = annotation.sample
        labels = [-1]*len(channel1)
        for j in range(len(samples)):
            qrs_region = self.get_qrs_region(samples, j, window_size)
            for sample in qrs_region:
                labels[sample] = 1
        return np.asarray(features), np.asarray(labels)

    def get_qrs_region(self, samples, location, window_size):
        if location < len(samples) - (int(window_size/2) + 1):
            return [q for q in range(samples[location] - (int(window_size/2)), samples[location] +
                                     int(window_size/2)+1)]
        else:
            gap = len(samples) - location - 1
            return [q for q in range(samples[location] - gap, samples[location]+gap)]

    def normalized_gradient(self, channel):
        gradient = np.diff(channel)
        gradient_norm = np.sqrt(np.sum(np.square(gradient)))
        normalized_gradient = np.divide(gradient, gradient_norm)
        return normalized_gradient

    def overwrite_signal(self, first_channel, second_channel, record):
        for i in range(len(record.p_signal)):
            record.p_signal[i][0] = first_channel[i]
            record.p_signal[i][1] = second_channel[i]
        return record

    def passband_filter(self, channel):
        freq = 360.0/2.0
        b, a = signal.butter(1, [5/freq, 12/freq], btype="band")
        new_channel = signal.filtfilt(b, a, channel)
        return new_channel

    def extract_from_all(self,split_size):
        features = []
        labels = []
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".dat"):
                name = signal_name.replace(".dat","")
                feature, label = self.extract_features('sample/'+name, split_size=split_size)
                for feat in feature:
                    features.append(feat)
                labels.extend(label)
        return np.asarray(features), np.asarray(labels)

