import wfdb
import numpy as np
from collections import defaultdict
from bisect import bisect_left
from rpeakdetection.Utility import Utility

ut = Utility()

class LabelsExtraction:

    def extract(self, include_vf=False, from_annot = False):
        print("Extracting labels")
        ann_path = '../../../data/ecg/mitdb/'
        labels = defaultdict(list)
        names_file = open("../../../data/mitdb_names.txt", "r")
        for line in names_file:
            name = line.replace("\n", '')
            if name == '207' and include_vf:
                annotation = wfdb.rdann(ann_path + name + '_VF', 'atr')
                ann_samples = annotation.sample
                ann_symbols = annotation.symbol
            else:
                ann_symbols = ut.remove_non_beat(ann_path + name)[1]
                ann_samples = ut.remove_non_beat(ann_path + name)[0]
            if from_annot:
                labels[name] = ann_symbols
            else:
                rpeak_output_dir = "../../../data/peaks/pantompkins/mitdb"
                input_samples_file = open(rpeak_output_dir + '/' + name + '.tsv', 'r')
                input_samples = []
                output_labels = []
                for line in input_samples_file:
                    sample_location = line.replace('\n', '')
                    input_samples.append(int(sample_location))
                for input in input_samples:
                    closest = self.take_closest(ann_samples, input)
                    if closest == len(ann_samples):
                        output_labels.append(ann_symbols[closest - 1])
                    else:
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