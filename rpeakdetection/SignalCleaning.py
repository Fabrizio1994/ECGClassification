import wfdb
import numpy as np
import os


class SignalCleaning:

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

    def remove_non_beat_for_all(self):
        for signal_name in os.listdir("data/sample/mitdb"):
            if signal_name.endswith(".atr"):
                name = signal_name.replace(".atr", "")
                new_ann, new_symbol = self.remove_non_beat("data/sample/mitdb/" + name)
                wfdb.wrann(name, "atr", np.asarray(new_ann), np.asarray(new_symbol))
                #os.system("mv " + signal_name + " annotations/beat")

    #returns an updated record object (useful for plotting)
    def overwrite_signal(self, first_channel, second_channel, record):
        for i in range(len(record.p_signal)):
            record.p_signal[i][0] = first_channel[i]
            record.p_signal[i][1] = second_channel[i]
        return record



