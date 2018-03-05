import wfdb
import numpy as np
import os
from GridSearch import GridSearch
from FeatureExtraction import FeatureExtraction

fe = FeatureExtraction()

class Utility:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        ,'"', '@']

    def clean_signal(self, sample_name):
        print(sample_name)
        record = wfdb.rdrecord(sample_name)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[0])
        samples, symbols = self.remove_non_beat(sample_name)
        new_sample, new_symbol = self.update_annotations(channel, samples, symbols)
        new_sample = np.asarray(new_sample)
        new_symbol = np.asarray(new_symbol)
        wfdb.wrann(sample_name.replace("sample/", ""), "atr", new_sample, new_symbol)

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
                annotated_loc = samples[j]
                qrs_region =[q for q in range(annotated_loc-5,annotated_loc+6)]
                qrs_values = [channel[samples[j]+q] for q in qrs_region]
                index_max = qrs_region[0]
                max = abs(qrs_values[0])
                for j in range(len(qrs_region)):
                    if abs(qrs_values[j]) > (max):
                        max = abs(qrs_values[j])
                        index_max = qrs_region[j]
                new_sample.append(index_max)
                new_symbol.append(symbols[j])
        return new_sample, new_symbol

    def clean_all(self):
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".atr"):
                self.clean_signal("sample/"+signal_name.replace(".atr",""))

    def remove_non_beat_for_all(self):
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".atr"):
                name = signal_name.replace(".atr","")
                new_sample, new_symbol = self.remove_non_beat("sample/"+name)
                wfdb.wrann(name,"atr", np.asarray(new_sample), np.asarray(new_symbol))

    def read_all(self):
        features = []
        labels = []
        for name in os.listdir("features"):
            print("reading signal "+name.replace(".tsv",""))
            file = open("features/"+name,"r")
            for line in file:
                vector = line.split("\t")
                features.append([float(vector[0]), float(vector[1])])
                labels.append(int(vector[2].replace("\n","")))
            file.close()
        return np.asarray(features), np.asarray(labels)

    def read_signal(self, name):
        features = []
        labels = []
        file = open("features/" + name, "r")
        for line in file:
            vector = line.split("\t")
            features.append([float(vector[0]), float(vector[1])])
            labels.append(int(vector[2].replace("\n", "")))
        return np.asarray(features), np.asarray(labels)

    def write_csv_signal(self, signal_name):
        file1 = open("csv/"+signal_name+"_1.csv", "w")
        file2 = open("csv/"+signal_name+"_2.csv", "w")
        record = wfdb.rdrecord("sample/"+signal_name)
        for elem in record.p_signal:
            file1.write("%s\n" % (str(elem[0])))
            file2.write("%s\n" % (str(elem[1])))
        file1.close()
        file2.close()

    def get_command(self, file_name):
        return "python2 RPeakDetection.py 360  < csv/" + file_name+ " > rpeak_output/" + file_name

    def run_rpeak(self):
        for file_name in os.listdir("csv"):
            os.system(self.get_command(file_name))

    def run_knn(self):
        for name in os.listdir("features"):
            signal_name = name.replace(".tsv", "")
            train_features, train_labels = fe.extract_features("sample/" + signal_name)
            GridSearch(signal_name, train_features, train_labels, ["not QRS", "QRS"])

