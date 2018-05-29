import wfdb
import numpy as np
from collections import defaultdict
from bisect import bisect_left

class LabelsExtraction:

    def extract(self, rpeak_output_dir, include_vf=False):

        labels = defaultdict(list)
        names_file = open("../../data/names.txt", "r")
        for line in names_file:
            name = line.replace("\n", '')
            print(name)
            annotation = wfdb.rdann('../../data/sample/mitdb/' + name, 'atr')
            if name == '207' and include_vf:
                annotation = wfdb.rdann('../../data/sample/mitdb/' + name + '_VF', 'atr')
            ann_symbols = annotation.symbol
            ann_samples = annotation.sample
            input_samples_file = open('../' + rpeak_output_dir + '/' + name + '.tsv', 'r')
            input_samples = []
            output_labels = []
            for line in input_samples_file:
                sample_location = line.replace('\n', '')
                input_samples.append(int(sample_location))
            closest = -1

            for input in input_samples:
                '''min = np.inf
                for ann_index in range(len(ann_samples)):
                    if ann_index < len(ann_samples - 1):
                        if abs(input - ann_samples[ann_index]) < min:
                            min = abs(input - ann_samples[ann_index])
                        else:
                            output_labels.append(ann_symbols[ann_index - 1])
                            break
                    else:
                        output_labels.append(ann_symbols[ann_index])
                        break'''
                closest = self.take_closest(ann_samples, input)
                if closest == len(ann_samples):
                    output_labels.append(ann_symbols[closest - 1])
                    break
                if ann_samples[closest] == ann_samples[-1]:
                    output_labels.append(ann_symbols[closest])
                    break
                output_labels.append(ann_symbols[closest])
            labels[name] = output_labels
        return labels

    def take_closest(self, annotation_samples, peak_location):
        pos = bisect_left(annotation_samples, peak_location)
        if pos == 0:
            return pos
        if pos == len(annotation_samples):
            return pos
        before = annotation_samples[-1]
        after = annotation_samples[pos]
        if after - peak_location < peak_location - before:
            return pos
        else:
            return pos - 1


if __name__ == '__main__':
    le = LabelsExtraction()
    le.extract('../data/peaks/pantompkins/mitdb')