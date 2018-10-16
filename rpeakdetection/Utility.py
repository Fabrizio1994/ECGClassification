import wfdb
import numpy as np
import os

# 100 min height R wave = 0.605 -> min_height
# Provare con minimo assoluto (tra tutti i segnali)
# Provare con minimo relativo (uno per ogni segnale)
# minimumRR = 72 samples
# peakutils.indexes ( ARRAY, thresh = min_height, min_dist = minimumRR)

class Utility:

    BEAT_ANN = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']

    def remove_non_beat(self, sample_name, rule_based):
        if rule_based:
            self.BEAT_ANN.extend(['[', '!', ']', '(BII\x00'])
        annotation = wfdb.rdann(sample_name, "atr")
        beat_ann = list()
        beat_sym = list()
        samples = annotation.sample
        symbols = annotation.symbol
        for j in range(len(annotation.sample)):
            if symbols[j] == '+' and rule_based:
                symbols[j] = annotation.aux_note[j]
            if symbols[j] in self.BEAT_ANN:
                symbol = symbols[j]
                peak = samples[j]
                beat_ann.append(peak)
                beat_sym.append(symbol)
        assert len(beat_ann) == len(beat_sym)
        return beat_ann, beat_sym

    def remove_non_beat_for_all(self, signals_dir, rule_based):
        symbols = dict()
        peaks = dict()
        for name in wfdb.get_record_list('mitdb'):
            new_peaks, new_symbol = self.remove_non_beat(signals_dir + name, rule_based)
            symbols[name] = new_symbol
            peaks[name] = new_peaks
        return peaks, symbols

    def write_annotation_peaks_file(self, file_names_path):
        file = open(file_names_path, "r")
        for line in file:
            name = line.replace("\n", "")
            ann = wfdb.rdann("../../../data/ecg/mitdb/" + name, 'atr')
            ann_file = open("../../../data/peaks/annotations/" + name + ".tsv", "w")
            for loc in ann.sample:
                ann_file.write("%s\n" % str(loc))
            ann_file.close()





