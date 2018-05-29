import wfdb
import os
from beatclassification.LabelsExtraction import LabelsExtraction
from pandas_ml import ConfusionMatrix


class Evaluation:

    """
    
        Parameters
        ----------
         category_map : map{str : list(str)}
            A map containing the name of the category as key and the annotation related to such a category as value.

        Returns
        -------
        The method initializes an evaluation map that has categories name as key and a map as value. This map
        is constituted by three values: TP, FP, FN all initialized to zero.

    """

    def initialize_map(self, category_map):
        evaluation_map = {}
        for symbol in category_map.keys():
            evaluation_map[symbol] = {}
            evaluation_map[symbol]['TP'] = 0
            evaluation_map[symbol]['FP'] = 0
            evaluation_map[symbol]['FN'] = 0
        evaluation_map['N'] = {}
        evaluation_map['N']['TP'] = 0
        evaluation_map['N']['FP'] = 0
        evaluation_map['N']['FN'] = 0
        return evaluation_map

    """
    
        Parameters
        ----------
         original_symbols : list(str)
            A list of strings containing the beat symbols of a certain signal.
         aux_symbols: list(str)
            A list of strings containing the aux symbols of a certain signal.

        Returns
        -------
        The method returns a list of strings representing the symbols adjusted for our purposes, that is, only 
        containing the following values : PVC, BII, N, VF. 

    """
    def clean_symbols(self, original_symbols, aux_symbols):
        cleaned_symbols = []
        non_beat_annotation = ['x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', 's', 'T', '*', 'D', '=', '"',
                               '@']

        for j in range(len(original_symbols)):
            if original_symbols[j] == '+' and aux_symbols[j] == '(BII\x00':
                original_symbols[j] = aux_symbols[j]
        original_symbols = list(filter(lambda x: x not in non_beat_annotation, original_symbols))
        for k in range(len(original_symbols)):
            if original_symbols[k] == '(BII\x00':
                cleaned_symbols.append('BII')
                continue
            if original_symbols[k] in ['[', '!', ']']:
                cleaned_symbols.append('VF')
                continue
            if original_symbols[k] == 'V':
                cleaned_symbols.append('PVC')
                continue
            if original_symbols[k] not in ['+', '[', '!', ']', 'BII', 'V']:
                cleaned_symbols.append('N')
                continue

        return cleaned_symbols[2:len(cleaned_symbols)]

    def clean_annotations(self, annotations):
        cleaned_annotations = []
        non_beat_annotation = ['x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', 's', 'T', '*', 'D', '=', '"',
                               '@', '+']

        #TODO: Remember to remove '+' from non beat annotations in order to capture BII

        annotations = list(filter(lambda x: x not in non_beat_annotation, annotations))
        for k in range(len(annotations)):
            if annotations[k] in ['[', '!', ']']:
                cleaned_annotations.append('VF')
                continue
            if annotations[k] == 'V':
                cleaned_annotations.append('PVC')
                continue
            if annotations[k] not in ['+', '[', '!', ']', 'BII', 'V']:
                cleaned_annotations.append('N')
                continue

        return cleaned_annotations[2:len(cleaned_annotations) - 1]

    """
    
        Parameters
        ----------
         database : str
            The database name. For the moment just 'mitdb' or 'incartdb'
          
        Returns
        -------
        The method creates two files in output called sensitivity and precision that respectively contain the value of 
        sensitivity and precision for each signal with respect to the categories we are working on.

    """
    def eval_rr_intervals(self, database, approach):

        le = LabelsExtraction()
        annotations = le.extract('../data/peaks/pantompkins/mitdb', True)
        category = {'PVC': ['V'],
                    'VF': ['[', '!', ']'],
                    'BII': ['BII'],
                    'N': ['N']}
        sensitivity_file = open('../../data/results/' + database + '/' + approach + '_sensitivity.tsv', 'a')
        precision_file = open('../../data/results/' + database + '/' + approach + '_precision.tsv', 'a')
        sensitivity_file.write("|patient|")
        precision_file.write("|patient|")
        for cat in sorted(category.keys()):
            sensitivity_file.write("%s|" % cat)
            precision_file.write("%s|" % cat)
        sensitivity_file.write("\n")
        precision_file.write("\n")
        names_file = open('../../data/names.txt', 'r')
        for line in names_file:
            predictions = []
            patient_name = line.replace('\n', '')
            #annotations = wfdb.rdann('database/' + database + '/original_annotations/' + patient, 'atr')
            file = open('../../data/labels/' + approach + '/' + database + '/' + patient_name + '.tsv', 'r')
            for value in file:
                predictions.append(value.replace('\n', ''))
            evaluation_map = self.initialize_map(category)
            #symbols = annotations.symbol
            #aux_symbols = annotations.aux_note
            #cleaned_symbols = self.clean_symbols(symbols, aux_symbols)
            cleaned_annotations = self.clean_annotations(annotations[patient_name])
            print(patient_name)
            #evaluation_map = self.evaluate_prediction(cleaned_symbols, evaluation_map, predictions)
            evaluation_map = self.evaluate_prediction(cleaned_annotations, evaluation_map, predictions)
            sensitivity_file.write('|%s|' % patient_name)
            precision_file.write('|%s|' % patient_name)
            for categ in sorted(evaluation_map.keys()):
                tp = evaluation_map[categ]['TP']
                fn = evaluation_map[categ]['FN']
                fp = evaluation_map[categ]['FP']
                if tp == 0 and fn == 0:
                    se = 'null'
                else:
                    se = tp / (tp + fn)
                if tp == 0 and fp == 0:
                    prec = 'null'
                else:
                    prec = tp / (tp + fp)
                if se != 'null':
                    se = round(se, 2)
                if prec != 'null':
                    prec = round(prec, 2)
                sensitivity_file.write('%s|' % str(se))
                precision_file.write('%s|' % str(prec))
            sensitivity_file.write('\n')
            precision_file.write('\n')

    """
    
        Parameters
        ----------
         cleaned_symbols : list(str)
            A list of strings containing the labels of a certain signal cleaned by the clean_symbols method.
         evaluation : map {str: {str : int}}
            A map that has a string representing an arrhythmia category as key and a map as value. This map has a string
            representing one of the field we are computing, that are TP, FP, FN, as key and an integer as value that
            stands for the number of times we have classificated a label in that category.
         predictions: list(str)
            A list of strings containing the labels of a certain signal obtained with the Tsipouras rule-based method.
            
        Returns
        -------
        The method evaluates each single prediction and returns a map containing the whole evaluation for a patient.

    """
    def evaluate_prediction(self, cleaned_symbols, evaluation_map, predictions):
        j = 0
        k = 0
        while j < (len(cleaned_symbols) - 1) and k < (len(predictions) - 1):
            label = cleaned_symbols[j]
            pred = predictions[k]
            if label == pred:
                evaluation_map[pred]['TP'] += 1
                j += 1
                k += 1
            else:
                evaluation_map[pred]['FP'] += 1
                evaluation_map[label]['FN'] += 1
                j += 1
                k += 1
        return evaluation_map


if __name__ == '__main__':
    eval = Evaluation()
    eval.eval_rr_intervals('mitdb', 'pantompkins')