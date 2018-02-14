import wfdb


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
