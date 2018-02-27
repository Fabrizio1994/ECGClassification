import wfdb
from scipy import signal
import numpy as np
import os

class FeatureExtraction:
    def extract_features(self, sample_name):
        print("Extracting features for signal " + sample_name + "...")
        record = wfdb.rdrecord(sample_name)
        annotation = wfdb.rdann(sample_name, 'atr', return_label_elements=["symbol","label_store","description"])
        first_channel = []
        second_channel = []
        for elem in record.p_signal:
            first_channel.append(elem[0])
            second_channel.append(elem[1])
        filtered_first_channel = self.passband_filter(first_channel)
        filtered_second_channel = self.passband_filter(second_channel)
        #record = self.overwrite_signal(filtered_first_channel, filtered_second_channel, record)
        #wfdb.plot_wfdb(record, annotation=annotation, time_units="seconds")
        gradient_channel1 = self.normalized_gradient(filtered_first_channel)
        gradient_channel2 = self.normalized_gradient(filtered_second_channel)
        print("actual peaks:"+ str(len(annotation.sample)))

        return self.compute_features(sample_name, gradient_channel1, gradient_channel2, annotation)

    def compute_features(self, sample_name, channel1, channel2, annotation):
        file = open("features/" + sample_name.replace("sample/","") + ".tsv", "w")
        features = []
        labels = []
        for i in range(len(channel1)):
            features.append([channel1[i], channel2[i]])
            if i in annotation.sample:
                labels.append(1)
                file.write("%s\t%s\t%s\n" % (channel1[i], channel2[i], str(1)))
            else:
                labels.append(-1)
                file.write("%s\t%s\t%s\n" % (channel1[i], channel2[i], str(-1)))
        return np.asarray(features), np.asarray(labels)

    def normalized_gradient(self, channel):
        gradient = np.diff(channel)
        #gradient = np.gradient(channel)
        gradient_norm = np.sqrt(np.sum(np.square(gradient)))
        #gradient_norm = max(gradient)
        normalized = np.divide(gradient, gradient_norm)
        return normalized

    def overwrite_signal(self, first_channel, second_channel, record):
        for i in range(len(record.p_signal)):
            record.p_signal[i][0] = first_channel[i]
            record.p_signal[i][1] = second_channel[i]
        return record

    def passband_filter(self, channel):
        freq = 360.0/2.0
        #b,a = signal.butter(1,[5/freq, 12/freq], btype="band")
        b, a = signal.butter(2, 11/freq, btype='lowpass')
        d, c = signal.butter(1, 5/freq, btype='highpass')
        #new_channel = signal.lfilter(d, c, signal.lfilter(b, a, channel))
        new_channel = signal.filtfilt(d,c, signal.filtfilt(b, a, channel))
        #new_channel = signal.filtfilt(b,a, channel)
        return new_channel

    def func_filter(self, channel):
        d = [-1, 32, 1]
        c = [1, 1]
        b = [1, -2, 1]
        a = [1, -2, 1]
        #new_channel = signal.filtfilt(b, a, signal.filtfilt(d, c, channel))
        new_channel = signal.lfilter(b, a, signal.lfilter(d, c, channel))
        return new_channel

    def extract_from_all(self):
        features = []
        labels = []
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".dat"):
                name = signal_name.replace(".dat","")
                feature, label = self.extract_features('sample/'+name)
                for feat in feature:
                    features.append(feat)
                labels.extend(label)
        return np.asarray(features), np.asarray(labels)

