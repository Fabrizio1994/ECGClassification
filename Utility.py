import wfdb
import numpy as np
import os
from GridSearch import GridSearch
from FeatureExtraction import FeatureExtraction
from KNN import KNN
from RPeakEvaluation import RPeakEvaluation

SIG_LEN = 650000

rpe = RPeakEvaluation()
knn = KNN()
fe = FeatureExtraction()
gs = GridSearch()

class Utility:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        , '"', '@']
    CLEANING_WINDOW = 10
    WINDOW_SIZES = [10, 20, 50]
    ANNOTATION_TYPES = ['beat', 'cleaned']

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

    def write_csv_signal(self, signal_name):
        file = open("csv/" + signal_name + ".csv", "w")
        record = wfdb.rdrecord("sample/" + signal_name)
        for elem in record.p_signal:
            file.write("%s\n" % (str(elem[0])))
        file.close()

    def get_command(self, file_name):
        return "python2 RPeakDetection.py 360  < csv/" + file_name+ " > rpeak_output/" + file_name

    def run_rpeak(self):
        for file_name in os.listdir("csv"):
            os.system(self.get_command(file_name))

    def run_knn(self):
        for name in os.listdir("sample"):
            if name.endswith('.atr'):
                signal_name = name.replace(".atr", "")
                for type in self.ANNOTATION_TYPES:
                    annotation = wfdb.rdann('annotations/' + type + '/' + signal_name, 'atr')
                    for size in self.WINDOW_SIZES:
                        train_features, train_labels = fe.extract_features("sample/" + signal_name, type, size)
                        prediction = gs.grid_search(train_features, train_labels)
                        cleaned_prediction = knn.clean_prediction(prediction)
                        locations = knn.get_index(cleaned_prediction)
                        labels = rpe.get_labels(annotation.sample, size)
                        rpe.evaluate_prediction(locations, labels, name, SIG_LEN, size, type, classifier="KNN")
