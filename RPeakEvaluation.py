import os
from KNN import KNN
from FeatureExtraction import FeatureExtraction
from Utility import Utility
import wfdb

ut = Utility()
fe = FeatureExtraction()
knn = KNN()


class RPeakEvaluation:

    SIZE_LAST_20 = 130000
    SIG_LEN = 650000

    def validate_r_peak(self):
        for name in os.listdir("features"):
            if name.endswith('.tsv'):
                signame = name.replace(".tsv", "")
                print(signame)
                annotation = wfdb.rdann("sample/"+signame, 'atr')
                locations = list(filter(lambda x: x > (self.SIG_LEN - self.SIZE_LAST_20), annotation.sample))
                prediction1 = self.get_predictions(signame, 0)
                prediction2 = self.get_predictions(signame, 1)
                self.evaluate_prediction(prediction1, locations, signame, 1, self.SIZE_LAST_20)
                self.evaluate_prediction(prediction2, locations, signame, 2, self.SIZE_LAST_20)

    def get_predictions(self, signame, n_channel):
        record = wfdb.rdrecord('sample/' + signame)
        channel = []
        window = 20
        for elem in record.p_signal:
            channel.append(elem[n_channel])
        prediction = []
        file = open("rpeak_output/" + str(signame) + "_"+str(n_channel+1)+".csv", "r")
        for line in file:
            value = int(line.replace("\n", ""))
            if value > self.SIG_LEN - self.SIZE_LAST_20:
                real_peak_index = self.get_r_peak(channel, value, window)
                prediction.append(real_peak_index)
        return prediction

    def get_r_peak(self, channel, value, window):
        indexes = range(int(value-window/2), int(value+window/2))
        max = channel[value]
        index_max = value
        for index in indexes:
            if channel[index] > max:
                max = channel[index]
                index_max = index
        return index_max

    def evaluate_prediction(self, prediction, locations, signame, channel_number, length):
        TP = 0
        FP = 0
        FN = 0
        window = 5
        i = 0
        j = 0
        while i < len(prediction) and j < len(locations):
            if prediction[i] == locations[j]:
                TP += 1
                i += 1
                j += 1
            else:
                found = False
                for w in range(1, window + 1):
                    if (prediction[i] + w) == locations[j] or (prediction[i] - w) == locations[j] and not found:
                        TP += 1
                        i += 1
                        j += 1
                        found = True
                if not found:
                    if prediction[i] > locations[j]:
                        FN += 1
                        j += 1
                    else:
                        FP += 1
                        i += 1
        FN += len(locations) - j
        FP += len(prediction) - i

        TN = length - TP - FP - FN
        file = open("report_rpeak.tsv", "a")
        file.write("%s_%s\n" %(signame, channel_number))
        file.write("TP:%s\tTN:%s\tFP:%s\tFN:%s\n" % (str(TP), str(TN), str(FP), str(FN)))
