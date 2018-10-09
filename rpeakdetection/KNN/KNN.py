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
import time
import math
rpd = RPeakDetector()
ut = Utility()
eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:

    DB = "mitdb"

    def rpeak_detection(self,names=None, train=True):
        window_sizes = [100]
        channels = [[0]]
        test_sizes = [0.97]
        min_dists = [70]
        recalls = list()
        precisions = list()
        result = dict()
        if names is None:
            names = wfdb.get_record_list('mitdb')
        # we use the first 54s for training and the remaining 97% of the signal for testing
        for name in names :
            for window_size in window_sizes:
                for channel in channels:
                    path = ("data/ecg/"+self.DB+"/"+name)
                    rpeak_locations = ut.remove_non_beat(path, rule_based=False)[0]
                    record, X, Y = fe.extract_features(name=name,path=path, rpeak_locations=rpeak_locations,
                                                       window_size=window_size,
                                                        write=False, channels=channel)
                    for test_size in test_sizes:
                        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=False,
                                                                            test_size=test_size)
                        if train:
                            self.start_time, predicted = gs.predict(X_train, X_test, y_train, name)
                        else:
                            loaded_model = pickle.load(open('rpeakdetection/KNN/classifiers/knn_'+name+'.pkl', 'rb'))
                            self.start_time = time.time()
                            predicted = loaded_model.predict(X_test)
                        test_index = len(X_train)*window_size
                        for min_dist in min_dists:
                            peaks = self.get_peaks(predicted, window_size, record, test_index, min_dist)
                            result[name] = peaks
                            recall, precision = rpd.evaluate(peaks, path, window_size, False, test_index)
                            recalls.append(recall)
                            precisions.append(precision)
                            print('{:s}, {:f}, {:f}'.format(name, precision, recall))
        av = 'average'
        print("{:s}, {:f}, {:f}".format(av, np.mean(precisions), np.mean(recalls)))
        return result




    def get_peaks(self, predicted_regions, window_size, record, test_index, min_dist):
        # we use always the first channel for taking the maximum in the qrs region
        signal = record[0]
        # a minimum distance of 40 samples is considered between two Rpeaks due to physiological reasons
        # min_dist = 40
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
        print("elapsed time for a single sample:")
        print(elapsed_time/len(signal))
        return Y_predicted

    def plot(self, region, neighbors, critic, labels):
        critic_index = critic % len(region)
        f, axarr = plt.subplots(len(neighbors) + 1, sharex=True)
        plot_x = np.arange(len(region))
        axarr[0].plot(plot_x, region)
        axarr[0].scatter([critic_index], [region[critic_index]], color ='red', label = 'detected_peak')
        axarr[0].set_title('test_region')
        axarr[0].legend()
        for i in range(1, len(neighbors)+1):
            axarr[i].set_title('neighbor')
            axarr[i].plot(plot_x, neighbors[i-1])
            if labels[i-1] == 1:
                peak = np.argmax(np.abs(neighbors[i-1]))
                axarr[i].scatter([peak], neighbors[i-1][peak], color = 'blue', label = 'real_peak')
                axarr[i].legend()
        plt.savefig('neighbors.png')


    def get_critic_detection(self):
        real_peaks = ut.remove_non_beat('data/ecg/mitdb/' + name, False)[0]
        signal, X, Y = fe.extract_features(name, 'data/ecg/mitdb/' + name, real_peaks, window_size, channels=channels,
                                           write=False)
        signal = signal[0]
        peaks = knn.rpeak_detection([name], False)[name]
        return X, Y, pd.plot_criticism(signal, name, peaks)[0]

    def get_k_neighbors(self,X , Y, critic, window_size):
        region_index = int(math.modf(critic / window_size)[1])
        region = X[region_index]
        model = pickle.load(open('rpeakdetection/KNN/classifiers/knn_' + name + '.pkl', 'rb'))
        n_neighbors = model.get_params()['n_neighbors']
        _, indexes = model.kneighbors([region], n_neighbors)
        indexes = indexes.flatten()
        neighbors = [X[i] for i in indexes]
        labels = [Y[i] for i in indexes]
        return  region, neighbors, labels


if __name__ == '__main__':
    from rpeakdetection.generics.peak_detector import PeakDetector
    knn = KNN()
    pd = PeakDetector()
    name = '102'
    channels = [0]
    window_size = 100
    test_size = 0.97
    X, Y, critic = knn.get_critic_detection()
    region, neighbors, labels = knn.get_k_neighbors(X, Y, critic, window_size)
    knn.plot(region, neighbors, critic, labels)