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
            signame = name.replace(".tsv","")
            print(signame)
            annotation = wfdb.rdann("sample/"+signame,'atr')
            locations = list(filter(lambda x: x > (self.SIG_LEN - self.SIZE_LAST_20), annotation.sample))
            prediction1 = self.get_predictions(signame,1)
            prediction2 = self.get_predictions(signame,2)
            self.evaluate_prediction(prediction1, locations, signame, 1, self.SIZE_LAST_20)
            self.evaluate_prediction(prediction2, locations, signame, 2, self.SIZE_LAST_20)

    def get_predictions(self, signame, n_channel):
        prediction = []
        file = open("rpeak_output/" + str(signame) + "_"+str(n_channel)+".csv", "r")
        for line in file:
            value = int(line.replace("\n", ""))
            if value > self.SIG_LEN - self.SIZE_LAST_20:
                prediction.append(int(line.replace("\n", "")))
        return prediction

    def evaluate_prediction(self, prediction, locations, signame, channel_number, length):
        TP = 0
        FP = 0
        FN = 0
        for pred in prediction:
            if pred in locations:
                TP +=1
            else:
                FP += 1
        for loc in locations:
            if loc not in prediction:
                FN += 1
        TN = length - TP - FP - FN
        file = open("report_rpeak.tsv", "a")
        file.write("%s_%s\n" %(signame, channel_number))
        file.write("TP:%s\tTN:%s\tFP:%s\tFN:%s\n" % (str(TP), str(TN), str(FP), str(FN)))
