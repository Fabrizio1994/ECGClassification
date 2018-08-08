import wfdb
import numpy as np
import os


class Utility:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        , '"', '@']

    def remove_non_beat(self, sample_name):
        annotation = wfdb.rdann(sample_name, "atr")
        non_beat_ann = []
        non_beat_sym = []
        samples = annotation.sample
        symbols = annotation.symbol
        for j in range(len(annotation.sample)):
            if symbols[j] not in self.NON_BEAT_ANN:
                non_beat_ann.append(samples[j])
                non_beat_sym.append(symbols[j])
        return non_beat_ann, non_beat_sym

    def remove_non_beat_for_all(self, signals_dir):
        for signal_name in os.listdir(signals_dir):
            if signal_name.endswith(".atr"):
                name = signal_name.replace(".atr", "")
                new_ann, new_symbol = self.remove_non_beat(signals_dir + name)
                wfdb.wrann(name, "atr", np.asarray(new_ann), np.asarray(new_symbol))

    def download_dataset(self, URL, names_path):
        names_file = open(names_path, "r")
        for line in names_file:
            name = line.replace("\n", "")
            os.system("curl -L " + URL + name + ".hea >> " + name + ".hea")
            os.system("curl -L " + URL + name + ".dat >> " + name + ".dat")
            os.system("curl -L " + URL + name + ".atr >> " + name + ".atr")

    def write_annotation_peaks_file(self, file_names_path):
        file = open(file_names_path, "r")
        for line in file:
            name = line.replace("\n", "")
            ann = wfdb.rdann("../../../data/ecg/mitdb/" + name, 'atr')
            ann_file = open("../../../data/peaks/annotations/" + name + ".tsv", "w")
            for loc in ann.sample:
                ann_file.write("%s\n" % str(loc))
            ann_file.close()





