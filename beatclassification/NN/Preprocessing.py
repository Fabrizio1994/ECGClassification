from beatclassification.NN.BeatExtraction import BeatExtraction
from beatclassification.LabelsExtraction import LabelsExtraction
from beatclassification.NN.keras.LSTM import LSTM_NN
import matplotlib.pyplot as plt
import numpy as np
import sys
beat_extraction = BeatExtraction()
labels_extraction = LabelsExtraction()
lstm = LSTM_NN()

val_dataset = [ "103",  "121",  "210",
                "221", "222", "228",  "233", "234"]

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

    def extract_database(self, Y, dataset, one_hot, windows, window_size, channels, next_beat):
        X = list()
        labels = list()
        for name in dataset:
            if windows:
                if next_beat:
                    symbols = Y[name][window_size-2:-1]
                else:
                    symbols = Y[name][window_size-1:-window_size]
            else:
                symbols = Y[name][1:-1]
            signal = beat_extraction.extract(name, window_size, channels)
            filtered_X, filtered_Y = self.filter(symbols, signal, one_hot)
            X.extend(filtered_X)
            labels.extend(filtered_Y)
        # plt.show()
        # sys.exit()
        return X, labels

    # exclude beats not in N, S, V, F, Q classes
    def filter(self, symbols, signal, one_hot):
        filtered_X = list()
        filtered_Y = list()
        for i in range(len(symbols)):
            if symbols[i] in list(symbol2class.keys()):
                filtered_X.append(signal[i])
                classe = symbol2class[symbols[i]]
                label = self.class2label[classe]
                if one_hot :
                    one_hot = [0]*len(classes)
                    one_hot[label] = 1
                    filtered_Y.append(one_hot)
                else:
                    filtered_Y.append(label)
        return filtered_X, filtered_Y

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

    def assign_weights(self, Y_train):
        weights = []
        # labels not in one-hot format
        for i in range(5):
            number_of_instances = len(list(filter(lambda y: y[i] == 1, Y_train)))
            weights.append(number_of_instances)
        weights = [1 / w for w in [q / sum(weights) for q in weights]]
        class_weigths = {}
        for i in range(5):
            class_weigths[i] = weights[i]
        return class_weigths

    def preprocess(self, scale_factors=None, weights=False, one_hot=True, windows=False, window_size=None,
                   channels=[0], next_beat=False):
        self.classes_ids()
        class_weights = None
        Y = labels_extraction.extract(from_annot=True)
        print("Extracting beats")
        X_train, Y_train = self.extract_database(Y, train_dataset, one_hot, windows, window_size, channels,
                                                 next_beat)
        X_val, Y_val = self.extract_database(Y, val_dataset, one_hot, windows, window_size, channels,
                                             next_beat)
        X_test, Y_test = self.extract_database(Y, test_dataset, one_hot, windows, window_size, channels
                                               , next_beat)
        '''print("Original train  distribution")
        for i in range(len(classes)):
            print(str(i) + " : " + str(len(list(filter(lambda x : np.argmax(x) == i, Y_train)))))'''
        if scale_factors is not None:
            X_train, Y_train = self.resample(X_train, Y_train, scale_factors)
            # print("Scaled train distribution")
            # for i in range(len(classes)):
            #    print(str(i) + " : " + str(len(list(filter(lambda x : np.argmax(x) == i, Y_train)))))
        #print("Test distribution:")
        #for i in range(len(classes)):
        #    print(str(i) + " : " + str(len(list(filter(lambda x : np.argmax(x) == i, Y_test)))))
        if weights:
            class_weights = self.assign_weights(Y_train)
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test),\
                np.array(X_val), np.array(Y_val),class_weights

    def classes_ids(self):
        # label is the integer value associated to a class
        self.class2label = {}
        count = 0
        for classe in classes:
            self.class2label[classe] = count
            count += 1





