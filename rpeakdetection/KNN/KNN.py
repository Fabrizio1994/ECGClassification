import wfdb
import numpy as np
from rpeakdetection.KNN.GridSearch import GridSearch
from rpeakdetection.KNN.FeatureExtraction import FeatureExtraction
from rpeakdetection.Evaluation import Evaluation
from rpeakdetection.Utility import Utility
from rpeakdetection.rpeak_detector import RPeakDetector
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split


rpd = RPeakDetector()
ut = Utility()
eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    WINDOW_SIZE= 32
    DB = "mitdb"

    def rpeak_detection(self):
        av_precisions = list()
        av_recalls = list()
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        test_sizes = [0.99, 0.97, 0.95, 0.93, 0.91]
        feature_group = 'window'
        for name in wfdb.get_record_list('mitdb'):
            path = ("data/ecg/"+self.DB+"/"+name)
            rpeak_locations = ut.remove_non_beat(path)[0]
            record, X, Y = fe.extract_features(signal_name=path,rpeak_locations=rpeak_locations,
                                                                       window_size=self.WINDOW_SIZE,
                                                                       feature_group=feature_group                                                                           )
            for t_size in test_sizes:
                X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False,
                                                                    test_size=t_size, random_state=42)
                predicted = gs.predict(X_train, X_test, y_train, name, t_size)
                print("n rpeaks predicted")
                print(len(list(filter(lambda x: x == 1, predicted))))
                test_index = len(X_train)*self.WINDOW_SIZE
                peaks = self.get_peaks(predicted, self.WINDOW_SIZE, record, test_index)
                print("test_size")
                print(t_size)
                recall, precision = rpd.evaluate(peaks, path, self.WINDOW_SIZE, test_index)
                precisions[t_size].append(precision)
                recalls[t_size].append(recall)
        for t_s in test_sizes:
            av_precisions.append(np.mean(precisions[t_s]))
            av_recalls.append(np.mean(precisions[t_s]))
        plt.plot(test_sizes, av_precisions, label="precision")
        plt.plot(test_sizes, av_recalls, label="recall")
        plt.savefig("rpeakdetection/KNN/reports/test_sizes_pr.png")
        plt.close()



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
                peak_loc = window_start + np.argmax(qrs_region)
                if peak_loc - prev > min_dist:
                    Y_predicted.append(peak_loc)
                prev = peak_loc
            window_start += window_size
        return Y_predicted

if __name__ == '__main__':
    knn = KNN()
    knn.rpeak_detection()






