import wfdb
import numpy as np
from collections import defaultdict
class LabelsExtraction:

    def extract(self, rpeak_output_dir, include_vf=False):

        labels = defaultdict(list)
        names_file = open("data/names.txt", "r")
        for line in names_file:
            name = line.replace("\n", '')
            print(name)
            annotation = wfdb.rdann('data/sample/mitdb/' + name, 'atr')
            if name == '207' and include_vf:
                annotation = wfdb.rdann('data/sample/mitdb/' + name + '_VF', 'atr')
            ann_symbols = annotation.symbol
            ann_samples = annotation.sample
            input_samples_file = open(rpeak_output_dir + '/' + name + '.tsv', 'r')
            input_samples = []
            output_labels = []
            for line in input_samples_file:
                sample_location = line.replace('\n', '')
                input_samples.append(int(sample_location))
            for input in input_samples:
                min = np.inf
                for ann_index in range(len(ann_samples)):
                    if abs(input - ann_samples[ann_index]) < min:
                        min = abs(input - ann_samples[ann_index])
                    else:
                        output_labels.append(ann_symbols[ann_index - 1])
                        break
            labels[name] = output_labels
        return labels