import wfdb
import numpy as np
import os
from subprocess import call

class Utility:

    NON_BEAT_ANN = [ '[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D', '='
        ,'"', '@']

    # CODE TO CLEAN SIGNAL FROM NON BEAT ANNOTATIONS
    def clean_signal(self, sample_name):
        print(sample_name)
        annotation = wfdb.rdann(sample_name, "atr")
        record = wfdb.rdrecord(sample_name)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[0])
        window = 5
        samples = annotation.sample.tolist()
        symbols = annotation.symbol
        new_sample, new_symbol = self.update_annotations(channel, samples, symbols, window)
        diff = self.check_convergence(new_sample, samples)
        while diff != 0:
            samples = new_sample
            new_sample, new_symbol = self.update_annotations(channel, samples, symbols, window)
            diff = self.check_convergence(new_sample, samples)
        print(diff)
        new_sample = np.asarray(new_sample)
        new_symbol = np.asarray(new_symbol)
        wfdb.wrann(sample_name.replace("sample/", ""), "atr", new_sample, new_symbol)

    def check_convergence(self, new_sample, samples):
        diff = 0
        for j in range(len(samples)):
            if samples[j] != new_sample[j]:
                diff += 1
                print(samples[j], new_sample[j])
        return diff

    def update_annotations(self, channel, samples, symbols, window):
        new_sample = []
        new_symbol = []
        for j in range(len(samples)):
            if symbols[j] not in self.NON_BEAT_ANN:
                max = abs(channel[samples[j]])
                index_max = samples[j]
                for w in range(1, window + 1):
                    if abs(channel[samples[j] + w]) > max:
                        max = abs(channel[samples[j] + w])
                        index_max = samples[j] + w
                    elif abs(channel[samples[j] - w]) > max:
                        max = abs(channel[samples[j] - w])
                        index_max = samples[j] - w
                new_sample.append(index_max)
                new_symbol.append(symbols[j])
        return new_sample, new_symbol

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
