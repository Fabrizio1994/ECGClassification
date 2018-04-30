import wfdb
import numpy as np

ann_path = "../data/sample/mitdb"


class FeatureExtraction:
    def __init__(self):
        # classes of beats
        self.classes = ["N", "S", "V", "F"]
        # physiobank symbols for beats
        self.symbols = ["N", "L", "R", "e", "J", "A", "a", "J", "S", "V", "E", "F"]
        # associates symbols to classes of beats
        self.symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                             "A": "S", "a": "S", "J": "S", "S": "S",
                             "V": "V", "E": "V",
                             "F": "F"}
        # integer values associated to the classes
        self.class2label = {}
        self.label2class = {}
        count = 0
        for classe in self.classes:
            self.class2label[classe] = count
            self.label2class[count] = classe
            count += 1

    def extract(self, names):
        features = []
        labels = []
        for name in names:
            print(name)
            annotation = wfdb.rdann(ann_path + "/" + name, "atr")
            symbols = annotation.symbol
            peaks = annotation.sample
            rr_intervals = np.diff(peaks)
            rr_mean = np.mean(rr_intervals)
            for count in range(5, len(symbols)-5):
                symbol = symbols[count]
                if symbol in self.symbols:
                    feature = []
                    window = rr_intervals[count - 5 : count + 5]
                    win_mean = np.mean(window)
                    prev_3 = rr_intervals[count - 3 : count]
                    prev_3 = np.divide(prev_3, rr_mean)
                    prev = rr_intervals[count - 1]
                    next = rr_intervals[count]
                    feature.extend([prev, next, win_mean])
                    feature.extend(prev_3)
                    features.append(feature)
                    classe = self.symbol2class[symbol]
                    label = self.class2label[classe]
                    labels.append(label)
        return np.array(features), np.array(labels)


