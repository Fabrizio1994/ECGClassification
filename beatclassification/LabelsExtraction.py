import wfdb
import numpy as np
from collections import defaultdict
from bisect import bisect_left
from rpeakdetection.Utility import Utility

ut = Utility()


class LabelsExtraction:

    def extract(self, ann_path, db='mitdb', peaks=None, include_vf=False, from_annot=True):
        """reads beat labels for each signal in a Physionet database
        :arg ann_path: path of the local folder containing the annotations files(.atr)
        :arg db: string identifier for the Physionet DB
        :arg include_vf: whether to include Ventricular Fibrillation(VF) annotations
        :arg peaks: list or numpy array containing the peaks locations. Used only if from_annot=False
        :arg from_annot: whether to associate labels to peaks from the ground truth(.atr file)
        :returns labels: a dictionary [signal_name, labels]
        """
        labels = defaultdict(list)
        if peaks is None:
            peaks = defaultdict(list)
        names = wfdb.get_record_list(db)
        for name in names:
            if name == '207' and include_vf:
                annotation = wfdb.rdann(ann_path + name + '_VF', 'atr')
                ann_symbols = annotation.symbol
            else:
                ann_samples, ann_symbols = ut.remove_non_beat(ann_path + name)
            if from_annot:
                peaks[name] = ann_samples
                labels[name] = ann_symbols
            else:
                output_labels = list()
                for peak in peaks:
                    closest = self.take_closest(ann_samples, peak)
                    if closest == len(ann_samples):
                        output_labels.append(ann_symbols[closest - 1])
                    else:
                        output_labels.append(ann_symbols[closest])
                labels[name] = output_labels
        return labels, peaks

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