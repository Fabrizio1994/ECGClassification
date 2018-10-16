import wfdb
import numpy as np
from rpeakdetection.KNN.GridSearch import GridSearch
from rpeakdetection.KNN.FeatureExtraction import FeatureExtraction
from rpeakdetection.Evaluation import Evaluation
from rpeakdetection.Utility import Utility
from rpeakdetection.rpeak_detector import RPeakDetector
import matplotlib.pyplot as plt
import pickle
import itertools
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import math
rpd = RPeakDetector()
ut = Utility()
eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:

    DB = "mitdb"

    def rpeak_detection(self,names=None):
        window_size = 100
        test_size = 0.97
        min_dist = 72
        approach_f = ['KNN_s']
        channels_f = ['1','12']
        filtered_f = ['RS']
        results = defaultdict(list)
        combinations = [approach_f, channels_f, filtered_f]
        if names is None:
            names = wfdb.get_record_list('mitdb')
        for comb in itertools.product(*combinations):
            recalls = list()
            precisions = list()
            times = list()
            print(comb)
        # we use the first 54s for training and the remaining 97% of the signal for testing
            for name in names :
                path = ("data/ecg/"+self.DB+"/"+name)
                rpeak_locations = ut.remove_non_beat(path, rule_based=False)[0]
                record, X, Y = fe.extract_features(name=name,path=path, rpeak_locations=rpeak_locations,
                                                   features_comb=comb)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False,
                                                                    test_size=test_size)
                train = not('KNN_w' in comb and '1' in comb and 'FS' in comb)
                if train:
                    self.start_time, predicted = gs.predict(X_train, X_test, y_train, name, comb)
                else:
                    loaded_model = pickle.load(open('rpeakdetection/KNN/classifiers/knn_'+name+'.pkl', 'rb'))
                    self.start_time = time.time()
                    predicted = loaded_model.predict(X_test)
                if 'KNN_s' in comb:
                    elapsed_time = time.time() - self.start_time
                    elapsed_time = elapsed_time / len(record[0])
                    test_index = len(X_train)
                    peaks = self.get_sample_peaks(predicted, test_index, min_dist)
                if 'KNN_w' in comb:
                    test_index = len(X_train) * window_size
                    elapsed_time, peaks = self.get_peaks(predicted, window_size, record, test_index, min_dist)
                recall, precision = rpd.evaluate(peaks, path, window_size, False, test_index)
                recalls.append(recall)
                precisions.append(precision)
                times.append(elapsed_time)
            comb_name = comb[0] + '_' + comb[1] + '_' + comb[2]
            results[comb_name] = [np.mean(precisions), np.mean(recalls), np.mean(times)]
            print("{:s}, {:f}, {:f}, {:f}".format(comb_name, np.mean(precisions), np.mean(recalls), np.mean(times)))
        print(results)
        with open('results.pkl', 'wb') as fid:
            pickle.dump(results, fid)

    def get_sample_peaks(self, predicted, test_index, min_dist):
        all_peaks = np.where(predicted == 1)
        all_peaks = list(map(lambda x: x + test_index, all_peaks))
        prev = 0
        peaks = list()
        all_peaks = np.array(all_peaks).flatten().tolist()
        for p in all_peaks:
            if p - prev > min_dist:
                peaks.append(p)
                prev = p
        return peaks

    def get_peaks(self, predicted_regions, window_size, record, test_index, min_dist):
        # we use always the first channel for taking the maximum in the qrs region
        signal = record[0]
        Y_predicted = list()
        window_start = test_index
        prev = 0
        for label in predicted_regions:
            if label == 1:
                window_end = window_start + window_size
                qrs_region = [abs(signal[value]) for value in range(window_start, window_end)]
                rpeak_loc = window_start + np.argmax(qrs_region)
                if rpeak_loc - prev > min_dist:
                    Y_predicted.append(rpeak_loc)
                prev = rpeak_loc
            window_start += window_size
        elapsed_time = time.time() - self.start_time
        #print("elapsed time for a single sample:")
        elapsed_time = elapsed_time/len(signal)
        return elapsed_time, Y_predicted


if __name__ == '__main__':
    knn = KNN()
    knn.rpeak_detection()