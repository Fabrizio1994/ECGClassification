import wfdb
import numpy as np

peaks_path = "../../../data/peaks/annotations/"
BEAT_SIZE = 340
train_dataset = ["101", "106", "108","109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203",
                 "205", "207", "208", "209", "215", "220", "223", "230"]
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
window_size = 3


class BeatExtraction:
    def extract(self):
        print("Extracting Beats:")
        X_train = self.extract_db(train_dataset)
        X_test = self.extract_db(test_dataset)
        X_train = self.compute_windows(X_train)
        X_test = self.compute_windows(X_test)
        return X_train, X_test

    def read_peaks(self, path):
        values = []
        file = open(path, "r")
        for line in file:
            values.append(line.replace("\n", ""))
        return values

    def extract_signal(self, record):
        signal = [[],[]]
        for elem in record.p_signal:
            signal[0].append(elem[0])
            signal[1].append(elem[1])
        return signal

    def extract_db(self, database):
        X = []
        for name in database:
            print(name)
            record = wfdb.rdrecord("../../../data/sample/mitdb/" + name)
            # extract signals from record
            signal = self.extract_signal(record)
            peaks = [int(v) for v in self.read_peaks(peaks_path + name + ".tsv")]
            # REMOVE FIRST AND LAST BEAT due to underflow and overflow
            for peak in peaks[1:-1]:
                beat = []
                beat.append(signal[0][peak - int(BEAT_SIZE / 2):peak + int(BEAT_SIZE / 2)])
                #beat.append(signal[1][peak - int(BEAT_SIZE / 2):peak + int(BEAT_SIZE / 2)])
                beat = [item for sublist in beat for item in sublist]
                X.append(beat)
        return X

    def compute_windows(self, X):
        windows = []
        for q in range(len(X) - window_size + 1):
            beat = [X[q] for q in range(q, q + window_size)]
            windows.append(beat)
        return np.array(windows)