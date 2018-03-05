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

    def get_predictions(self, signame, n_channel, total_size=SIG_LEN,
                        test_size=SIZE_LAST_20):
        record = wfdb.rdrecord('sample/' + signame)
        channel = []
        window = 10
        for elem in record.p_signal:
            channel.append(elem[n_channel])
        prediction = []
        file = open("rpeak_output/" + str(signame) + "_"+str(n_channel+1)+".csv", "r")
        for line in file:
            value = int(line.replace("\n", ""))
            if value > total_size - test_size:
                real_peak_index = self.get_r_peak(channel, value, window)
                prediction.append(real_peak_index)
        return prediction

    def get_r_peak(self, channel, value, window):
        indexes = range(int(value-window/2), int(value+window/2+1))
        max = abs(channel[value])
        rpeak = value
        for index in indexes:
            if abs(channel[index]) > max:
                max = channel[index]
                rpeak = index
        return rpeak

    def evaluate_prediction(self, prediction, locations, signame, channel_number, length):
        TP = 0
        FP = 0
        FN = 0

        i = 0
        j = 0
        while i < len(prediction) and j < len(locations):
            qrs_region = [q for q in range(locations[j] - 5, locations[j] + 6)]
            if prediction[i] in qrs_region:
                TP += 1
                i += 1
                j += 1
            elif prediction[i] > locations[j]:
                FN += 1
                j += 1
            else:
                FP += 1
                i += 1

        FN += len(locations) - j
        FP += len(prediction) - i

        TN = length - TP - FP - FN
        file = open("report_"+str(channel_number)+".tsv", "a")
        file.write("%s\n" %(signame))
        file.write("TP:%s\tTN:%s\tFP:%s\tFN:%s\n" % (str(TP), str(TN), str(FP), str(FN)))
