import wfdb
import numpy as np
from rpeakdetection.KNN.GridSearch import GridSearch
from rpeakdetection.KNN.FeatureExtraction import FeatureExtraction
from rpeakdetection.Evaluation import Evaluation
from rpeakdetection.Utility import Utility
from rpeakdetection.rpeak_detector import RPeakDetector
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split


rpd = RPeakDetector()
ut = Utility()
eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    # we segment the signal in windows of 36 samples == av QRS width
    WINDOW_SIZE= 36
    DB = "mitdb"

    def rpeak_detection(self, train=True):
        # we use the first 54s for training and the remaining 97% of the signal for testing
        test_size = 0.97
        for name in ['228', '230', '231', '232', '233', '234']:
            path = ("data/ecg/"+self.DB+"/"+name)
            rpeak_locations = ut.remove_non_beat(path, rule_based=False)[0]
            record, X, Y = fe.extract_features(name=name,path=path, rpeak_locations=rpeak_locations,
                                               window_size=self.WINDOW_SIZE,
                                           write=True)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False,
                                                                test_size=test_size, random_state=42)
            if train:
                predicted = gs.predict(X_train, X_test, y_train, name, test_size)
            else:
                loaded_model = pickle.load(open('rpeakdetection/KNN/classifiers/knn_'+name+'_'+str(test_size)+'.pkl', 'rb'))
                predicted = [loaded_model.predict(x) for x in X_test]
            print("n rpeaks predicted")
            print(len(list(filter(lambda x: x == 1, predicted))))
            test_index = len(X_train)*self.WINDOW_SIZE
            peaks = self.get_peaks(predicted, self.WINDOW_SIZE, record, test_index)
            recall, precision = rpd.evaluate(peaks, path, self.WINDOW_SIZE, test_index)
            result = open('rpeakdetection/KNN/reports/signal_pr.txt','a')
            result.write('%s\t%s\t%s\n' % (name, precision, recall))

    def get_peaks(self, predicted_regions, window_size, record, test_index):
        # we use always the first channel for taking the maximum in the qrs region
        signal = record[0]
        # a minimum distance of 40 samples is considered between two Rpeaks due to physiological reasons
        min_dist = 40
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
        return Y_predicted


if __name__ == '__main__':
    knn = KNN()
    knn.rpeak_detection()






