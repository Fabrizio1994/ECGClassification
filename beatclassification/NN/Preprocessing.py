from beatclassification.NN.BeatExtraction import BeatExtraction
from beatclassification.LabelsExtraction import LabelsExtraction
import matplotlib.pyplot as plt
import numpy as np
import sys

beat_extraction = BeatExtraction()
labels_extraction = LabelsExtraction()

train_dataset = ['106', '112', '122', '201', '223', '230',"108","109", "115", "116", "118", "119", "124",
                 "205", "207", "208", "209", "215",'101', '114','203','220']
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
classes = ["N", "S", "V", "F", 'Q']
# associates symbols to classes of beats
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                     "A": "S", "a": "S", "J": "S", "S": "S",
                     "V": "V", "E": "V",
                     "F": "F",
                '/': 'Q', 'f' : 'Q', 'Q':'Q'}


class Preprocessing:

    def extract_database(self, Y, dataset, one_hot, windows, window_size, channels, from_annot, ann_path, peaks):
        X = list()
        labels = list()
        for name in dataset:
            print(name)
            signal_peaks = peaks[name]
            symbols = Y[name][1:-1]
            if windows:
                symbols = symbols[window_size-1:-window_size]
            signal_X = beat_extraction.extract(name, window_size, from_annot, ann_path, signal_peaks, channels)
            signal_Y = self.assign_labels(symbols, one_hot)
            labels.extend(signal_Y)
            X.extend(signal_X)
            assert len(signal_X) == len(signal_Y)
        return X, labels

    def assign_labels(self, symbols, one_hot):
        signal_Y = list()
        for i in range(len(symbols)):
            symbol = symbols[i]
            if symbol in list(symbol2class.keys()):
                classe = symbol2class[symbol]
                label = self.class2label[classe]
            else:
                label= self.class2label['N']
            if one_hot :
                one_hot = [0]*len(classes)
                one_hot[label] = 1
                signal_Y.append(one_hot)
            else:
                signal_Y.append(label)
        return signal_Y

    def resample(self, X_train, Y_train, scale_factors):
        X = []
        Y = []
        k = 0
        for j in range(len(X_train)):
            label = np.argmax(Y_train[j])
            # under sampling
            if scale_factors[label] < 0:
                if k == abs(scale_factors[label]):
                    X.extend([X_train[j]])
                    Y.extend([Y_train[j]])
                    k = 0
                else:
                    k += 1
            else:
                X.extend([X_train[j]]*scale_factors[label])
                Y.extend([Y_train[j]] * scale_factors[label])
        return X, Y

    def preprocess(self, peaks=None, scale_factors=None, one_hot=True, windows=False, window_size=None,
                   channels=[0], from_annot=True, ann_path='data/ecg/mitdb/'):
        self.classes_ids()
        Y, peaks = labels_extraction.extract(from_annot=from_annot, ann_path=ann_path, peaks=peaks)
        print("Extracting beats")
        X_train, Y_train = self.extract_database(Y, train_dataset, one_hot, windows, window_size, channels, from_annot,
                                                 ann_path, peaks)
        X_test, Y_test = self.extract_database(Y, test_dataset, one_hot, windows, window_size, channels, from_annot,
                                                ann_path, peaks)
        if scale_factors is not None:
            X_train, Y_train = self.resample(X_train, Y_train, scale_factors)
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    def classes_ids(self):
        # label is the integer value associated to a class
        self.class2label = {}
        count = 0
        for classe in classes:
            self.class2label[classe] = count
            count += 1





