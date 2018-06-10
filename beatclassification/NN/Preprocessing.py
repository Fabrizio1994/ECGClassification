from beatclassification.NN.BeatExtraction import BeatExtraction
from beatclassification.LabelsExtraction import LabelsExtraction
from beatclassification.NN.keras.LSTM import LSTM_NN
import numpy as np
beat_extraction = BeatExtraction()
labels_extraction = LabelsExtraction()
lstm = LSTM_NN()

train_dataset = ["106", "108","109", "112", "115", "116", "118", "119", "122", "124", "201",
                 "205", "207", "208", "209", "215", "223", "230",'101', '114','203','220']
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]

supraventricular_db = [i for i in range(800,813)]
train_dataset.extend([str(e) for e in supraventricular_db])

classes = ["N", "S", "V", "F"]
# associates symbols to classes of beats
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                     "A": "S", "a": "S", "J": "S", "S": "S",
                     "V": "V", "E": "V",
                     "F": "F"}
class Preprocessing:

    # considers any symbol that is not in this focus of study as 'N'
    def filter_symbols(self, symbols):
        for j in range(len(symbols)):
            if symbols[j] not in symbol2class.keys():
                symbols[j] = 'N'
        return symbols

    def resample(self, X_train, Y_train, scale_factors):
        X = []
        Y = []
        k = 0
        for j in range(len(X_train)):
            label = np.argmax(Y_train[j])
            # undersampling
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
        return np.array(X), np.array(Y)

    def one_hot_labels(self, name, all_symbols):
        symbols = all_symbols[name]
        symbols = self.filter_symbols(symbols)
        classes = list(map(lambda x: symbol2class[x], symbols))
        labels = list(map(lambda x: self.class2label[x], classes))
        for index in range(len(labels)):
            one_hot = [0] * 4
            label = labels[index]
            one_hot[label] = 1
            labels[index] = one_hot
        return labels

    def assign_weights(self, Y_train):
        weights = []
        # labels not in one-hot format
        for i in range(4):
            number_of_instances = len(list(filter(lambda y: y[i] == 1, Y_train)))
            weights.append(number_of_instances)
        weights = [1 / w for w in [q / sum(weights) for q in weights]]
        class_weigths = {}
        print("class weights:")
        for i in range(4):
            class_weigths[i] = weights[i]
            print(weights[i])
        return class_weigths

    def preprocess_labels(self, all_symbols, window_size):
        Y_train = []
        Y_test = []
        for name in train_dataset:
            labels = self.one_hot_labels(name, all_symbols)
            # REMOVE FIRST AND LAST LABEL because of the overflow and underflow due to the window around the peak
            Y_train.append(labels[3:-3])
        for name in test_dataset:
            labels = self.one_hot_labels(name, all_symbols)
            Y_test.append(labels[3:-3])
        # all signals labels are flattened in one vector
        # some labels are excluded due to window size
        Y_train = np.array([item for sublist in Y_train for item in sublist])[window_size - 1:]
        Y_test = np.array([item for sublist in Y_test for item in sublist])[window_size - 1:]
        return Y_test, Y_train

    # scale = vector of scaling factors. A negative factor stands for undersampling
    def extract(self, scale, window_size):
        # label is the integer value associated to a class
        self.class2label = {}
        count = 0
        for classe in classes:
            self.class2label[classe] = count
            count += 1
        X_train, X_test = beat_extraction.extract(window_size)
        all_symbols = labels_extraction.extract(from_annot=True)
        Y_test, Y_train = self.preprocess_labels(all_symbols, window_size)
        print("original distribution")
        for i in range(4):
            number_of_instances = len(list(filter(lambda y: y[i] == 1, Y_train)))
            print(str(i) + " " + str(number_of_instances))
        X_train, Y_train = self.resample(X_train, Y_train, scale)
        print("scaled distribution")
        for i in range(4):
            number_of_instances = len(list(filter(lambda y: y[i] == 1, Y_train)))
            print(str(i) + " " + str(number_of_instances))
        return X_train, Y_train, X_test, Y_test#, X_val, Y_val



