import numpy as np
import wfdb
import os
from beatclassification.rule_based.Evaluation import Evaluation


class Main:
    """
        Parameters
        ----------
         time : float
            A float value that represent the time.

        Returns
        -------
        The method converts the time value (float) into a sample value (int).

    """
    def time2sample(self, time):
        # mitdb = 360
        # incartdb = 257
        return round(time * 360)

    """
        Parameters
        ----------
         rr_intervals : list(int)
            A list of strings representing the rr_intervals value for a certain signal
         index: int
            An integer value representing the index of the sliding window we are moving on. It indicates the middle RR 
            interval of the window, that contains three RR intervals.

        Returns
        -------
        The method updates the values of the current window moving them to the desired index.  

    """
    def update_window(self, rr_intervals, index):
            RR1 = rr_intervals[index - 1]
            RR2 = rr_intervals[index]
            RR3 = rr_intervals[index + 1]
            return RR1, RR2, RR3

    """
        Parameters
        ----------
         rr_interval_file : file
            A file containing the rr_intervals value for a certain signal separated by \n.
         patient: str
            A string value that represents the patient we want to work on.
         database: str
            The database name. For the moment just 'mitdb' or 'incartdb'
         approach: str
            The chosen approach. Form the moment just 'annotation' or 'pantompkins' or 'rpeak'

        Returns
        -------
        The method updates the values of the current window moving them to the desired index. Then, it writes an 
        output file, containing a label assigned to each RR-interval.

    """
    def find_beat_annotation(self, rr_interval_file, patient, database, approach):
        print(patient)
        const1 = 1.15
        const2 = 1.8
        const3 = 1.2
        rr_intervals = []
        # excluding first interval
        current_index = 1
        prediction = []

        for rr_interval in rr_interval_file:
            rr_interval = rr_interval.replace('\n', '')
            rr_intervals.append(int(rr_interval))

        # INITIALIZATION
        while current_index < len(rr_intervals) - 1:
            RR1, RR2, RR3 = self.update_window(rr_intervals, current_index)
            # RULE 1
            cond1 = RR2 < self.time2sample(0.6)
            cond2 = const2 * RR2 < RR1
            if cond1 and cond2:
                vf_prediction = ['VF']
                vf_index = current_index + 1
                if vf_index < len(rr_intervals) - 1:
                    RR1, RR2, RR3 = self.update_window(rr_intervals, vf_index)
                    condition = self.vf_condition(RR1, RR2, RR3)
                    while condition and vf_index < len(rr_intervals) - 1:
                        vf_prediction.append('VF')
                        vf_index = vf_index + 1
                        if vf_index < len(rr_intervals) - 1:
                            RR1, RR2, RR3 = self.update_window(rr_intervals, vf_index)
                            condition = self.vf_condition(RR1, RR2, RR3)
                if len(vf_prediction) >= 4:
                    prediction.extend(vf_prediction)
                    current_index = vf_index
                    continue
                else:
                    # go back to current index
                    RR1, RR2, RR3 = self.update_window(rr_intervals, current_index)

            # RULE 2
            cond1 = const1 * RR2 < RR1
            cond2 = const1 * RR2 < RR3
            cond3 = abs(RR1 - RR2) < self.time2sample(0.3)
            cond4 = RR1 < self.time2sample(0.8)
            cond5 = RR2 < self.time2sample(0.8)
            cond6 = RR3 > const3 * np.mean([RR1, RR2])
            cond7 = abs(RR2 - RR3) < self.time2sample(0.3)
            cond8 = RR2 < self.time2sample(0.8)
            cond9 = RR3 < self.time2sample(0.8)
            cond10 = RR1 > const3 * np.mean([RR2, RR3])
            if (cond1 and cond2) or (cond3 and cond4 and cond5 and cond6) or (cond7 and cond8 and cond9 and cond10):
                prediction.append('PVC')
                current_index = current_index + 1
                continue

            # RULE 3
            cond1 = RR2 > self.time2sample(2.2)
            cond2 = RR2 < self.time2sample(3.0)
            cond3 = abs(RR1 - RR2) < self.time2sample(0.2)
            cond4 = abs(RR2 - RR3) < self.time2sample(0.2)

            if (cond1 and cond2) and (cond3 or cond4):
                prediction.append('BII')
            else:
                prediction.append('N')
            current_index = current_index + 1

        out_file = open('data/labels/' + approach + '/' + database + '/' + patient + '.tsv', 'w')
        for value in prediction:
            out_file.write(value + '\n')

    def vf_condition(self, RR1, RR2, RR3):
        cond1 = RR1 < self.time2sample(0.7)
        cond2 = RR2 < self.time2sample(0.7)
        cond3 = RR3 < self.time2sample(0.7)
        cond4 = RR1 + RR2 + RR3 < self.time2sample(1.7)
        return (cond1 and cond2 and cond3) or cond4

    """
    
        Parameters
        ----------
        database : str
            The database name. For the moment just 'mitdb' or 'incartdb'
        approach: str
            The chosen approach. Form the moment just 'annotation' or 'pantompkins' or 'rpeak'
        
        Returns
        -------
        The method creates tsv files containing labels of the RR intervals for each signal.
        
    """
    def write_labels(self, database, approach):
        names_file = open('data/names.txt', 'r')
        for line in names_file:
            patient_name = line.replace('\n', '')
            rr_interval_file = open('data/rr_intervals/' + approach + '/' + database + '/' + patient_name + '.tsv', 'r')
            self.find_beat_annotation(rr_interval_file, patient_name, database, approach)

    """
    
        Parameters
        ----------
        database : str
            The database name. For the moment just 'mitdb' or 'incartdb'
        approach: str
            The chosen approach. Form the moment just 'annotation' or 'pantompkins' or 'rpeak'
        
        Returns
        -------
        The method creates tsv files representing the RR intervals for each signal.
        
    """
    def write_rr(self, database, approach):
        names_file = open('data/names.txt', 'r')
        for line in names_file:
            patient_name = line.replace('\n', '')
            file = open('data/peaks/' + approach + '/' + database + '/' + patient_name + ".tsv", "r")
            peaks_locations = []
            for line in file:
                line = line.replace("\n", "")
                peaks_locations.append(int(line))
            rr_intervals = np.diff(peaks_locations)
            file_w = open('data/rr_intervals/' + approach + '/' + database + '/' + patient_name + ".tsv", "w")
            for rr_interval in rr_intervals:
                file_w.write("%s\n" % str(rr_interval))

    """
    
        Parameters
        ----------
        database : str
            The database name. For the moment just 'mitdb' or 'incartdb'
        approach: str
            The chosen approach. Form the moment just 'annotation' or 'pantompkins' or 'rpeak'
            
        Returns
        -------
        The method creates tsv files representing the location of R-Peaks for each signal.

    """
    def write_peaks(self, database, approach):
        for name in sorted(os.listdir('database/' + database + '/cleaned_annotations')):
            if name.endswith('.atr'):
                patient = name.replace('.atr', '')
                annotations = wfdb.rdann('database/' + database + '/cleaned_annotations/' + patient, 'atr')
                file = open('peaks/' + approach + '/' + database + '/' + patient + '.tsv', 'w')
                for val in annotations.sample:
                    file.write('%s\n' % str(val))

    """
    
       Parameters
       ----------
       sample_name : str
        The path to reach the patient annotation file. 
       NON_BEAT_ANN : str[] 
        Array of strings each of which represent a non beat annotation. 
        
       Returns
       -------
       The method returns two arrays containing the beat annotation locations and the beat annotation symbols.

    """
    def remove_non_beat(self, sample_name, NON_BEAT_ANN):
        annotation = wfdb.rdann(sample_name, "atr")
        beat_ann = []
        beat_sym = []
        samples = annotation.sample
        symbols = annotation.symbol
        for j in range(len(annotation.sample)):
            if symbols[j] not in NON_BEAT_ANN:
                beat_ann.append(samples[j])
                beat_sym.append(symbols[j])
        return beat_ann, beat_sym

    """
    
       Parameters
       ----------
       database : str
        The database name. For the moment just 'mitdb' or 'incartdb'       

       Returns
       -------
       The method creates new annotation file (.atr), cleaned from the non-beat annotations, for each signal.

    """
    def remove_non_beat_for_all(self, database):
        NON_BEAT_ANN = ['x', '(', ')', 'p', 't', 'u', '`', '\'', '^', '|', '~', '+', 's', 'T', '*', 'D',
                        '=', '"', '@', '[', ']']
        for signal_name in os.listdir('database/' + database + '/original_annotations'):
            if signal_name.endswith(".atr"):
                name = signal_name.replace(".atr", "")
                if name != 'I04' and name != 'I17' and name != 'I35' and name != 'I44' and name != 'I57' and \
                                name != 'I72' and name != 'I74':
                    print(name)
                    beat_ann, beat_symbol = self.remove_non_beat('database/' + database + '/original_annotations/' + name,
                                                                 NON_BEAT_ANN)
                    wfdb.wrann(name, 'atr',
                               sample=np.asarray(beat_ann), symbol=np.asarray(beat_symbol))

if __name__ == '__main__':
    m = Main()
    eval = Evaluation()
    #database = ['incartdb', 'mitdb']
    database = 'mitdb'
    #approach = ['annotation', 'pantompkins', 'rpeak']
    approach = 'pantompkins'
    #m.remove_non_beat_for_all(database)


    ### CALL THIS METHOD ONLY FOR ANNOTATION APPROACH ###
    #m.write_peaks(database, approach)



    m.write_rr(database, approach)
    m.write_labels(database, approach)
    eval.eval_rr_intervals(database, approach)
