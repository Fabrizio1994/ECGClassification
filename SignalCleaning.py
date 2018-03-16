import wfdb
import numpy as np
import os


class SignalCleaning:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        , '"', '@']
    CLEANING_WINDOW = 10

    def clean_signal(self, sample_name):
        print(sample_name)
        record = wfdb.rdrecord("sample/"+sample_name)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[0])
        annotation = wfdb.rdann("annotations/beat/"+sample_name, "atr")
        samples = annotation.sample
        symbols = annotation.symbol
        new_sample, new_symbol = self.update_annotations(channel, samples, symbols)
        new_sample = np.asarray(new_sample)
        new_symbol = np.asarray(new_symbol)
        wfdb.wrann(sample_name, "atr", new_sample, new_symbol)
        os.system("mv "+sample_name+".atr"+" annotations/cleaned")

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

    def update_annotations(self, channel, samples, symbols):
        new_sample = []
        new_symbol = []
        for j in range(len(samples)):
                annotated = samples[j]
                qrs_window = [q for q in range(-int(self.CLEANING_WINDOW / 2), int(self.CLEANING_WINDOW / 2) + 1)]
                qrs_values = [channel[annotated+q] for q in qrs_window]
                qrs_indexes = [annotated + q for q in qrs_window]
                index_max = qrs_indexes[0]
                max = abs(qrs_values[0])
                for j in range(len(qrs_window)):
                    if abs(qrs_values[j]) > max:
                        max = abs(qrs_values[j])
                        index_max = qrs_indexes[j]
                new_sample.append(index_max)
                new_symbol.append(symbols[j])
        return new_sample, new_symbol

    def clean_all(self):
        for signal_name in os.listdir("sample"):
            if signal_name.endswith('.atr'):
                self.clean_signal(signal_name.replace(".atr",""))

    def remove_non_beat_for_all(self):
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".atr"):
                name = signal_name.replace(".atr","")
                new_sample, new_symbol = self.remove_non_beat("sample/"+name)
                wfdb.wrann(name, "atr", np.asarray(new_sample), np.asarray(new_symbol))
                os.system("mv " + signal_name + " annotations/beat")

    #returns an updated record object (useful for plotting)
    def overwrite_signal(self, first_channel, second_channel, record):
        for i in range(len(record.p_signal)):
            record.p_signal[i][0] = first_channel[i]
            record.p_signal[i][1] = second_channel[i]
        return record



