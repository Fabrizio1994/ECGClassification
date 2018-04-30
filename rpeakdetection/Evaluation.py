import os
from rpeakdetection.FeatureExtraction import FeatureExtraction
import wfdb
import numpy as np

fe = FeatureExtraction()

class Evaluation:
    siglen = 650000
    test_index = 520000
    test_size = 130000
    DB = "mitdb"

    # noinspection PyTypeChecker
    def validate_r_peak(self):
        prediction_window_sizes = [10]
        evaluation_window_size = 10
        for name in sorted(os.listdir("sample/" + self.DB)):
            if name.endswith('.atr'):
                signame = name.replace(".atr", "")
                print(signame)
                annotation = wfdb.rdann("sample/" + self.DB + "/" + signame, 'atr')
                self.siglen = wfdb.rdrecord("sample/" + self.DB + "/" + signame).sig_len
                self.test_index = int(self.siglen/5)*4
                self.test_size = self.siglen - self.test_index
                locations = list(filter(lambda x: x > self.test_index, annotation.sample))
                for window_size in prediction_window_sizes:
                    prediction = self.get_predictions(signame, 0, prediction_window_size=window_size)
                    labels = self.get_labels(locations, evaluation_window_size)
                    self.evaluate_rpeak_prediction(prediction, labels, signame, window_size, locations)

    def get_predictions(self, signame, n_channel, prediction_window_size):
        record = wfdb.rdrecord('sample/'+self.DB + "/" + signame)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[n_channel])
        prediction = []
        file = open("rpeak_output/" + self.DB + "/" + str(signame) + "_1.csv", "r")
        for line in file:
            zero_crossing = int(line.replace("\n", ""))
            if zero_crossing >= self.test_index:
                real_peak_index = self.get_r_peak(channel, zero_crossing, prediction_window_size)
                prediction.append(real_peak_index)
        return prediction

    #returns the intervals around the annotation locations
    def get_labels(self, locations, evaluation_window_size):
        labels = []
        interval = [q for q in range(int(-evaluation_window_size / 2), int(evaluation_window_size / 2) + 1)]
        for loc in locations:
            labels.append([loc + q for q in interval])
        return labels

    def get_r_peak(self, channel, zero_crossing, window_size):
        # overflow
        if int(zero_crossing + window_size / 2) >= len(channel):
            indexes = range(int(zero_crossing - window_size /2), len(channel))
        else:
            indexes = range(int(zero_crossing - window_size / 2), int(zero_crossing + window_size / 2 + 1))
        max = abs(channel[zero_crossing])
        rpeak = zero_crossing
        for index in indexes:
            if abs(channel[index]) > max:
                max = channel[index]
                rpeak = index
        return rpeak

    def evaluate_rpeak_prediction(self, peaks, intervals, signame, prediction_window_size, ann_locations):
        fn, fp, tp, tn, correct_preds = self.confusion_matrix(intervals, peaks)
        if tp != 0:
            der = ((fp + fn) / tp)
            der = round(der, 3)
        else:
            der = np.infty
            der = round(der, 3)
        if tp + fn != 0:
            se = (tp / (tp + fn)) * 100
            se = round(se, 3)
        else:
            se = 0
        sens_file = open("20-50.tsv","a")
        diff = self.compute_average_diff(correct_preds, ann_locations)
        diff = round(diff, 3)
        sens_file.write("|%s|%s|%s|%s|\n" % (signame, str(der), str(se), str(diff)))

    def confusion_matrix(self, labels, prediction):
        # prediction = peaks
        # labels = intervals of evaluation_window_size around the annotation
        TP = 0
        FP = 0
        FN = 0
        correct_preds = []
        unique_list_of_labels = [item for sublist in labels for item in sublist]
        for pred in prediction:
            if pred in unique_list_of_labels:
                TP += 1
                correct_preds.append(pred)
            else:
                FP += 1
        for label in labels:
            found = False
            for pred in prediction:
                if pred in label:
                    found = True
            if not found:
                FN += 1
        TN = self.test_size - TP - FP - FN
        return FN, FP, TP, TN, correct_preds

    def compute_sensitivity(self, fp, fn, tp):
        if tp + fn != 0:
            se = (tp / (tp + fn)) * 100
        else:
            se = 0

        return se


    def compute_average_diff(self, correct_preds, locations):
        count = 0
        sum = 0
        for pred in correct_preds:
            count += 1
            diff = min([abs(pred - loc) for loc in locations])
            sum += diff
        if count != 0:
            return sum / count
        else:
            return np.infty
