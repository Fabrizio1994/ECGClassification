'''import os
import wfdb
import numpy as np
beat_annotation = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
for name in sorted(os.listdir("sample/incartdb")):
    if name.endswith(".atr"):
        new_symbol = []
        new_sample = []
        name = name.replace(".atr", "")
        print(name)
        annotation = wfdb.rdann("sample/incartdb/" + name, "atr")
        j = 0
        for symbol in annotation.symbol:
            if symbol in beat_annotation:
                if annotation.sample[j] > 0:
                    new_symbol.append(symbol)
                    new_sample.append(annotation.sample[j])
            j += 1
        new_sample = np.array(new_sample)
        new_symbol = np.array(new_symbol)
        wfdb.wrann(name, "atr", new_sample, new_symbol)'''
file = open("reports/KNN/incartdb/qrsdetection/sliding_onechannel.tsv","r")
output = open("reports/KNN/incartdb/qrsdetection/sliding_onechannel.tsv", "w")
for line in file:
    line = line.split("|")
    format_string = "|%s|"
    for i in range(2, len(line)-1):
        line[i] = str(round(float(line[i]), 3))
        format_string += "%s|"
    format_string += "\n"
    output.write(format_string %(line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9],
                 line[10]))