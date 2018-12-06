import wfdb
import numpy as np
from rpeakdetection.KNN.GridSearch import GridSearch
from rpeakdetection.KNN.FeatureExtraction import FeatureExtraction
from rpeakdetection.Evaluation import Evaluation
from rpeakdetection.Utility import Utility
import matplotlib.pyplot as plt
import pickle
import itertools
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import math
ut = Utility()
eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:

    DB = "mitdb"
    # names = ['103', '115', '222', '203', '108', '113', '209', '212', '123', '214', '114', '109', '124', '105', '231', '100', '230', '201', '106', '111', '202', '107', '232', '112', '213', '101', '102', '104', '205', '200', '116', '233', '220', '210', '207', '228', '119', '221', '117', '122', '219', '234', '118', '223', '208', '121', '215', '217']

    def rpeak_detection(self, window_size=None, test_size=None, names=None, combinations=None):
        if window_size is None:
            window_size = 50
        if test_size is None:
            test_size = 0.97
        min_dist = 72
        evaluation_window_size = 36
        approach_f = ['KNN_s', 'KNN_w']
        channels_f = ['1', '2', '12']
        filtered_f = ['FS', 'RS']
        results = defaultdict(list)
        if combinations is None:
            combinations = [approach_f, channels_f, filtered_f]
        if names is None:
            names = wfdb.get_record_list('mitdb')
        for comb in itertools.product(*combinations):
            print(comb)
            if 'KNN_w' in comb:
                precisions, recalls, times = self.QRS_KNN(comb, min_dist, names, test_size, window_size,
                                                          evaluation_window_size)
            else:
                precisions, recalls, times = self.SSK(comb, names, evaluation_window_size)
            comb_name = comb[0] + '_' + comb[1] + '_' + comb[2]
            results[comb_name] = [np.mean(precisions), np.mean(recalls), np.mean(times)]
            print("{:s}, {:f}, {:f}, {:f}".format(comb_name, np.mean(precisions), np.mean(recalls), np.mean(times)))
        print(results)
        return results

    def QRS_KNN(self, comb, min_dist, names, test_size, window_size, evaluation_window_size):
        recalls = list()
        precisions = list()
        times = list()
        for name in names:
            path = ("data/ecg/" + self.DB + "/" + name)
            rpeak_locations = ut.remove_non_beat(path, rule_based=False)[0]
            record, X, Y = fe.extract_features(name=name, path=path, rpeak_locations=rpeak_locations,
                                               features_comb=comb, window_size=window_size)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False,
                                                                    test_size=test_size)
            #if train:
            self.start_time, predicted = gs.predict(X_train, X_test, y_train, name, comb)
            '''else:
                class_name = comb[0] + '_' + comb[1] + '_' + comb[2]
                loaded_model = pickle.load(
                    open('rpeakdetection/KNN/classifiers/' + name + '_' + class_name + '.pkl', 'rb'))
                self.start_time = time.time()
                predicted = loaded_model.predict(X_test)'''
            test_index = len(X_train) * window_size
            elapsed_time, peaks = self.get_peaks(predicted, window_size, record, test_index, min_dist)
            recall, precision = eval.evaluate(peaks, path, evaluation_window_size, False, test_index)
            print(recall)
            print(precision)
            recalls.append(recall)
            precisions.append(precision)
            times.append(elapsed_time)
        return precisions, recalls, times

    def get_sample_peaks(self, predicted, record):
        predicted = np.where(predicted == 1)
        prev = 0
        signal = record[0]
        predicted = np.array(predicted).flatten().tolist()
        regions = list()
        region = list()
        indexes = np.where(np.diff(predicted)==1)
        regions_values = [predicted[i] for i in indexes[0]]
        for p in regions_values:
            if p - prev == 1:
                region.append(p)
            else:
                if prev!= 0:
                    regions.append(region)
                region = list([p])
            prev = p
        average = sum(list(map(lambda x: len(x), regions )))/len(regions)
        filtered_regions =  list(filter(lambda x : len(x) >= average, regions))
        peaks = [np.argmax(np.abs(signal[region[0]:region[-1]])) + region[0] for region in filtered_regions]
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

    def SSK(self, comb, names, evaluation_window_size):
        precisions= list()
        recalls = list()
        times = list()
        # training is performed on the whole signal 100
        train_path = ("data/ecg/" + self.DB + "/100")
        train_rpeak_locations = ut.remove_non_beat(train_path, rule_based=False)[0]
        record, X_train, y_train = fe.extract_features(name='100', path=train_path, rpeak_locations=train_rpeak_locations,
                                           features_comb=comb)
        #if train:
        model = gs.SSK_train(X_train, y_train, comb)
        #else:
            #model =  pickle.load(open('rpeakdetection/KNN/classifiers/SSK_'+comb[1]+'_'+comb[2]+'.pkl', 'rb'))
        # testing is performed on all the other signals
        for name in names[1:]:
            path = ("data/ecg/" + self.DB +'/'+ name)
            rpeak_locations = ut.remove_non_beat(path, rule_based=False)[0]
            record, X_test, Y_test = fe.extract_features(name=name, path=path, rpeak_locations=rpeak_locations,
                                                 features_comb=comb)
            start_time = time.time()
            predicted = model.predict(X_test)
            print(sum(predicted))
            peaks = self.get_sample_peaks(predicted, record)
            elapsed_time = time.time() - start_time
            elapsed_time = elapsed_time / len(record[0])
            recall, precision = eval.evaluate(peaks, path, evaluation_window_size, False)
            print(recall)
            print(precision)
            recalls.append(recall)
            precisions.append(precision)
            times.append(elapsed_time)
        return precisions, recalls, times

    def compare_window_sizes(self, sizes):
        combinations = [['KNN_w'], ['1'], ['FS']]
        comb_name = 'KNN_w_1_FS'
        precisions = list()
        recalls = list()
        for window_size in sizes:
            result = self.rpeak_detection(window_size=window_size, combinations=combinations)
            precision, recall, times = result[comb_name]
            precisions.append(precision)
            recalls.append(recall)
        plt.plot(sizes, precisions, label='precision')
        plt.plot(sizes, recalls, label='recall')
        plt.xlabel('window size(samples)')
        plt.ylabel('score')
        best_size = sizes[int(np.argmax([x+y for x,y in zip(precisions, recalls)]))]
        plt.plot([best_size, best_size], [0,1])
        plt.legend()
        plt.savefig('window_size_pr.png')

    def compare_test_size(self, sizes):
        combinations = [['KNN_w'], ['1'], ['FS']]
        comb_name = 'KNN_w_1_FS'
        precisions = list()
        recalls = list()
        for test_size in sizes:
            result = self.rpeak_detection(test_size=test_size, combinations=combinations)
            precision, recall, times = result[comb_name]
            precisions.append(precision)
            recalls.append(recall)
        train_sizes=[int((1-s)*100) for s in sizes]
        plt.plot(train_sizes, precisions, label='precision')
        plt.plot(train_sizes, recalls, label='recall')
        plt.xlabel('training size %')
        plt.ylabel('score')
        plt.legend()
        plt.savefig('training_size_pr.png')



