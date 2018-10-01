import os
from rpeakdetection.KNN.FeatureExtraction import FeatureExtraction
import wfdb
import numpy as np

fe = FeatureExtraction()

class Evaluation:
    siglen = 650000
    test_index = 520000
    test_size = 130000
    DB = "mitdb"

    # a prediction inside a range of size=evaluation_window_size is considered TP
    def validate_r_peak(self, prediction, output_file):
        evaluation_window_size = 10
        for name in sorted(os.listdir("data/ecg/" + self.DB)):
            if name.endswith('.atr'):
                signame = name.replace(".atr", "")
                print(signame)
                annotation = wfdb.rdann("data/ecg/" + self.DB + "/" + signame, 'atr')
                locations = annotation.sample
                labels = self.get_labels(locations, evaluation_window_size)
                self.evaluate_rpeak_prediction(prediction, labels, signame, locations, output_file)

    # returns the intervals around the annotation locations
    def get_labels(self, locations, evaluation_window_size):
        labels = []
        interval = [q for q in range(int(-evaluation_window_size / 2), int(evaluation_window_size / 2) + 1)]
        for loc in locations:
            labels.append([loc + q for q in interval])
        return labels

    def evaluate_rpeak_prediction(self, peaks, intervals, signame, ann_locations, output_file):
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
        sens_file = open(output_file,"a")
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
