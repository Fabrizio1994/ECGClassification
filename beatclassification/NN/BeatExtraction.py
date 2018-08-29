import wfdb
import numpy as np

peaks_path = "../../../data/peaks/annotations/"
train_dataset = ["106", "108","109", "112", "115", "116", "118", "119", "122", "124", "201",
                 "205", "207", "208", "209", "215", "223", "230",'101', '114','203','220']

val_dataset = ['106', '112', '122', '201', '223', '230']

test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
LEFT_WINDOW = 70
RIGHT_WINDOW = 100


class BeatExtraction:

    def read_peaks(self, path):
        values = list()
        file = open(path, "r")
        for line in file:
            values.append(line.replace("\n", ""))
        return values

    def extract(self, name, window_size, channels=[0]):
        X = list()
        record = wfdb.rdrecord("../../../data/ecg/mitdb/" + name, channels=channels)
        # extract signals from record
        signal = record.p_signal
        if len(channels) == 1:
            signal = signal.flatten()
        peaks = [int(v) for v in self.read_peaks(peaks_path + name + ".tsv")]
        # REMOVE FIRST AND LAST BEAT due to underflow and overflow
        for peak in peaks[1:-1]:
            if len(channels) > 1:
                beat = list()
                for channel in range(len(channels)):
                    for sample in range(peak - LEFT_WINDOW, peak + RIGHT_WINDOW):
                        beat.append(signal[sample][channel])
                X.append(beat)
            else:
                X.append(signal[peak - LEFT_WINDOW:peak + RIGHT_WINDOW].tolist())
        if window_size is not None:
            X = self.compute_windows(X, window_size)
        return X

    def compute_windows(self, X, window_size):
        windows = []
        for q in range(len(X) - window_size + 1):
            beat = [X[q] for q in range(q, q + window_size)]
            windows.append(beat)
        return np.array(windows)