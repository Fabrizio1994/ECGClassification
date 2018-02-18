import wfdb
from FeatureExtraction import FeatureExtraction

fe = FeatureExtraction()


class Utility:
    def write_signal(self, signal_name):
        record = wfdb.rdrecord("../ECGClassification/samples/" + signal_name)
        signal = []
        i = 0
        for value in record.p_signal:
            signal.append(value[0])
            i += 1
        file = open("csv/" + signal_name + ".csv", "w")
        file.write("timestamp,ecg_measurements\n")
        for elem in signal:
            file.write("%s\n" % (str(elem)))

    def load_signal(self, signal_name):
        signal = []
        f = open("csv/" + signal_name + ".csv", "r")
        for line in f:
            signal.append(line.replace("\n",""))
        return signal

    def plot_filtered_unfiltered(self,signal_name, record):
        record = wfdb.rdrecord("samples/" + signal_name, sampto=2000)
        wfdb.plot_wfdb(record, title="pre-filter")
        record.p_signal = fe.passband_filter(record)
        wfdb.plot_wfdb(record, title ="post-filter")

    def load_feature(self, sig_name):
        features = []
        labels = []
        file = open("features/"+sig_name+".tsv","r")
        for line in file:
            vector = line.split("\t")
            features.append([float(vector[0]),float(vector[1])])
            labels.append(int(vector[2].replace("\n","")))
        return features, labels

