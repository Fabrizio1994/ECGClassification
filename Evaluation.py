import os
from FeatureExtraction import FeatureExtraction
import wfdb
import numpy as np

fe = FeatureExtraction()


class Evaluation:
    SIZE_LAST_20 = 130000
    SIG_LEN = 650000
    TEST_INDEX = SIG_LEN - SIZE_LAST_20

    def validate_r_peak(self):
        window_sizes = [10, 20, 50]
        annotation_type = ['beat', 'cleaned']

        for name in os.listdir("sample"):
            if name.endswith('.atr'):
                signame = name.replace(".atr", "")
                print(signame)
                for type in annotation_type:
                    annotation = wfdb.rdann("annotations/" + type + '/'
                                            + signame, 'atr')
                    locations = list(filter(lambda x: x > self.TEST_INDEX,
                                            annotation.sample))
                    for size in window_sizes:
                        prediction = self.__get_predictions(signame, 0,
                                                          window_size=size)
                        labels = self.get_labels(locations, size)
                        self.evaluate_prediction(prediction, labels,
                                                 signame, self.SIZE_LAST_20,
                                                 locations, window_size=size,
                                                 annotation_type=type,
                                                 classifier="RPeakDetection")

    def __get_predictions(self, signame, n_channel, window_size,
                        total_size=SIG_LEN,
                        test_size=SIZE_LAST_20):
        record = wfdb.rdrecord('sample/' + signame)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[n_channel])
        prediction = []
        file = open("rpeak_output/" + str(signame) + "_" + str(n_channel + 1)
                    + ".csv", "r")
        for line in file:
            value = int(line.replace("\n", ""))
            if value > total_size - test_size:
                real_peak_index = self.__get_r_peak(channel, value, window_size)
                prediction.append(real_peak_index)
        return prediction

    def __get_r_peak(self, channel, value, window_size):
        indexes = range(int(value - window_size / 2), int(value + window_size / 2 + 1))
        max = abs(channel[value])
        rpeak = value
        for index in indexes:
            if abs(channel[index]) > max:
                max = channel[index]
                rpeak = index
        return rpeak

    def evaluate_prediction(self, prediction, labels, signame, length,
                            ann_locations, window_size, annotation_type,
                            features_type, classifier):
        FN, FP, TP, TN, correct_preds = self.__confusion_matrix(length, labels, prediction)
        if TP != 0:
            DER = ((FP + FN) / TP)
        else:
            DER = np.infty
        if TP + FN != 0:
            SE = (TP / (TP + FN)) * 100
        else:
            SE = 0
        self.__write_results(DER, FN, FP, SE, TN, TP, ann_locations, annotation_type, classifier, correct_preds,
                             features_type, signame, window_size)

    def __confusion_matrix(self, length, labels, prediction):
        TP = 0
        FP = 0
        FN = 0
        correct_preds = []
        for pred in prediction:
            if pred in labels:
                TP += 1
                correct_preds.append(pred)
            else:
                FP += 1
        for label in labels:
            if label not in prediction:
                FN += 1
        TN = length - TP - FP - FN
        return FN, FP, TP, TN, correct_preds

    def __write_results(self, DER, FN, FP, SE, TN, TP, ann_locations,
                        annotation_type, classifier, correct_preds,
                        features_type, signame, window_size):
        file = open("reports/" + classifier + "/" + annotation_type + "_"
                    + str(window_size) + "_" + features_type + ".tsv", "a")
        if classifier == "KNN":
            file.write("|%s|%s|%s|%s|%s|%s|%s|\n" % (signame, str(TP), str(TN),
                                                     str(FP), str(FN), str(DER),
                                                     str(SE)))
        else:
            DIFF = self.__compute_average_diff(correct_preds, ann_locations)
            file.write("|%s|%s|%s|%s|%s|%s|%s|%s|\n" % (signame, str(TP),
                                                        str(TN), str(FP),
                                                        str(FN), str(DER),
                                                        str(SE), str(DIFF)))

    def __compute_average_diff(self, correct_preds, locations):
        count = 0
        sum = 0
        for pred in correct_preds:
            count += 1
            diff = min([abs(pred - loc) for loc in locations])
            sum += diff
        return sum / count
