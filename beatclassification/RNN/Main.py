from beatclassification.RNN.BeatExtraction import BeatExtraction
from beatclassification.LabelsExtraction import LabelsExtraction
from beatclassification.RNN.RNN import RNN
import numpy as np

be = BeatExtraction()
le = LabelsExtraction()


peaks_dir = "../data/peaks/pantompkins/mitdb"
train_dataset = ["101", "106", "108","109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203",
                 "205", "207", "208", "209", "215", "220", "223", "230"]
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
classes = ["N", "S", "V", "F"]
# associates symbols to classes of beats
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                     "A": "S", "a": "S", "J": "S", "S": "S",
                     "V": "V", "E": "V",
                     "F": "F"}


# considers any symbol that is not in this focus of study as 'N'
def filter_symbols(symbols):
    for j in range(len(symbols)):
        if symbols[j] not in symbol2class.keys():
            symbols[j] = 'N'
    return symbols


def one_hot_labels(name, all_symbols):
    symbols = all_symbols[name]
    symbols = filter_symbols(symbols)
    classes = list(map(lambda x: symbol2class[x], symbols))
    labels = list(map(lambda x: class2label[x], classes))
    for index in range(len(labels)):
        one_hot = [0] * 4
        label = labels[index]
        one_hot[label] = 1
        labels[index] = one_hot
    return labels


# label is the integer value associated to a class
class2label = {}
count = 0
for classe in classes:
    class2label[classe] = count
    count += 1
X_train, X_test = be.extract()
all_symbols = le.extract(peaks_dir)
Y_train = []
Y_test = []

for name in train_dataset:
    labels = one_hot_labels(name, all_symbols)
    # REMOVE FIRST AND LAST LABEL
    Y_train.append(labels[1:-1])

for name in test_dataset:
    labels = one_hot_labels(name, all_symbols)
    Y_test.append(labels[1:-1])

Y_train = np.array([item for sublist in Y_train for item in sublist])
Y_test = np.array([item for sublist in Y_test for item in sublist])
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

rnn = RNN(X_train, X_test, Y_train, Y_test)


