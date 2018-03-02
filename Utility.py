import wfdb
import numpy as np
import os
from subprocess import call

class Utility:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        ,'"', '@']

    # CODE TO CLEAN SIGNAL FROM NON BEAT ANNOTATIONS
    def clean_signal(self, sample_name):
        annotation = wfdb.rdann(sample_name, "atr")
        record = wfdb.rdrecord(sample_name)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[0])
        window = 20

        new_sample = []
        new_symbol = []
        samples = annotation.sample
        symbols = annotation.symbol
        while samples != new_sample:
            annotation = wfdb.rdann(sample_name, 'atr')
            new_sample = []
            new_symbol = []
            samples = annotation.sample

            for j in range(len(samples)):
                if symbols[j] not in self.NON_BEAT_ANN:
                    if channel[samples[j]] > 0:
                        max = channel[samples[j]]
                        index_max = samples[j]
                        for w in range(1, window + 1):
                            if channel[samples[j] + w] > max:
                                max = channel[samples[j] + w]
                                index_max = samples[j] + w
                            elif channel[samples[j] - w] > max:
                                max = channel[samples[j] - w]
                                index_max = samples[j] - w
                        new_sample.append(index_max)
                        new_symbol.append(symbols[j])
                    else:
                        min = 0
                        index_min = 0
                        for w in range(1, window + 1):
                            if channel[samples[j] + w] < min:
                                min = channel[samples[j] + w]
                                index_min = samples[j] + w
                            elif channel[samples[j] - w] < min:
                                min = channel[samples[j] - w]
                                index_min = samples[j] - w
                        new_sample.append(index_min)
                        new_symbol.append(symbols[j])
            new_sample = np.asarray(new_sample)
            new_symbol = np.asarray(new_symbol)
            print(sample_name)
            wfdb.wrann(sample_name.replace("sample/", ""), "atr", new_sample, new_symbol)

    def clean_all(self):
        for signal_name in os.listdir("sample"):
            if signal_name.endswith(".atr"):
                self.clean_signal("sample/"+signal_name.replace(".atr",""))

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
