import pywt
import wfdb
from rpeakdetection.Utility import Utility
import numpy as np
from collections import defaultdict
import random
from sklearn.decomposition import PCA
from scipy import signal
from scipy.signal import medfilt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from beatclassification.SVM_weighted.FeatureExtraction import FeatureExtraction

fe = FeatureExtraction()
ut = Utility()
ecg_path = 'data/ecg/mitdb/'
# classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
symbol2class = {"N": "N", "L": "N", "e": "N", "j": "N", "R": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F"}
# , '/': 'Q', 'f': 'Q', 'Q': 'Q'}

sig_len = 650000


class Preprocessing():

    def preprocess( self, dataset_names, channels, model, classes=None, aami=True, one_hot=True, timesteps=None,
                    filtered=False, train_db='mitdb', fs=360 ):
        if classes is None:
            classes = [ 'N', 'S', 'V', 'F' ]
        # duration of a beat(s)
        time_len = 0.422
        beat_size = int(fs * time_len)
        X_shape = self.compute_shape(dataset_names)
        X_shape = (X_shape, beat_size * len(channels))
        X = np.empty(X_shape)
        if one_hot:
            Y = np.empty((X_shape[ 0 ], len(classes)))
        else:
            Y = np.empty(X_shape[ 0 ], dtype='int8')
        count = 0
        for name in dataset_names:
            beats, labels, peaks = self.extract_labeled_beats(aami, classes, name, one_hot, model=model,
                                                              channels=channels,
                                                              filtered=filtered, train_db=train_db)
            X[ count:count + len(peaks) ] = beats
            Y[ count:count + len(peaks) ] = labels
            count += len(peaks)
        X = np.array(X)
        Y = np.array(Y)
        if timesteps is not None:
            X, Y = self.compute_timesteps(X, Y, timesteps)
        return X, Y

    def extract_features( self, beats, peaks ):
        rr_intervals = np.diff(peaks)
        rr_mean = np.mean(rr_intervals)
        features = list()
        for i, p in enumerate(peaks[ 5:-5 ], start=5):
            feature = beats[ i ].tolist()
            feature = fe.rr_features(i, feature, rr_intervals, rr_mean)
            feature = fe.signal_cumulants(beats[ i ], feature)
            features.append(feature)
        return features

    def extract_labeled_beats( self, aami, classes, name, one_hot, filtered=False, channels=None, model=None,
                               train_db='mitdb', multiclass=True, window=None, left_window=70, right_window=100 ):
        if train_db == 'incartdb':
            ecg_path = 'data/ecg/incartdb/'
        else:
            ecg_path = 'data/ecg/mitdb/'
        if channels is None:
            channels = [0]
        print(name)
        if name != '114' and len(channels) == 1:
            record = wfdb.rdrecord(ecg_path + name, channels=[ 0 ])
        else:
            record = wfdb.rdrecord(ecg_path + name, channels=channels)
        peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
        record = np.transpose(record.p_signal)
        if filtered:
            for id in range(len(record)):
                record[ id ] = self.filter(record[ id ])
        peaks, symbols = self.exclude_out_of_range(peaks, symbols, window=window, left_window=left_window, right_window=right_window)
        labels = self.extract_labels(aami, classes, one_hot, symbols, multiclass)
        if model == 'LSTM':
            beats = self.extract_beats(channels, peaks, record, window=window, left_window=left_window, right_window=right_window)
        else:
            beats = self.image_beats(peaks, record)
        return beats, labels, peaks

    def extract_beats( self, channels, peaks, record, window=None, left_window=70, right_window=100 ):
        if window is not None:
            left_window = int(window/2)
            right_window = left_window
        beat_len = (left_window + right_window) * len(channels)
        beats_shape = (len(peaks), beat_len)
        beats = np.zeros(beats_shape)
        for i, p in enumerate(peaks):
            beat = list()
            for id in range(len(record)):
                beat.extend(record[id][p - left_window: p + right_window])
            beats[ i ] = beat
        return beats

    def compute_timesteps( self, X, Y, timesteps ):
        end_beats = timesteps - 1
        start_label = end_beats
        windowed = np.zeros((X.shape[ 0 ] - end_beats, timesteps, X.shape[ 1 ]))
        for i in range(len(X) - end_beats):
            window = [ X[ w ] for w in range(i, i + timesteps) ]
            windowed[ i ] = window
        labels = Y[ start_label: ]
        return windowed, labels

    def exclude_out_of_range( self, peaks, symbols, window=None, left_window=70, right_window=100):
        if window is not None:
            left_window = int(window/2)
            right_window = left_window
        # exclude beats at the end of the signal (padding?), 21 in total. Excludes also beats out of the scope of study
        pairs = list(filter(lambda x: left_window <= x[ 0 ] < sig_len - right_window,
                            zip(peaks, symbols)))
        peaks, symbols = zip(*pairs)
        return peaks, symbols

    def extract_labels( self, aami, classes, one_hot, symbols, multiclass=True ):
        symbols = list(symbols)
        labels = np.zeros(len(symbols), dtype='int8')
        if one_hot:
            if multiclass:
                labels = np.zeros((len(symbols), len(classes)))
            else:
                labels = np.zeros((len(symbols), 1))
            return self.one_hot_labels(labels, symbols, aami, classes, multiclass)
        for i in range(len(symbols)):
            # exclude 15 unclassifiable beats
            if symbols[ i ] in symbol2class.keys():
                if aami:
                    classe = symbol2class[ symbols[ i ] ]
                    if multiclass:
                        label = classes.index(classe)
                        labels[ i ] = label
                    else:
                        if classe == 'N':
                            labels[ i ] = 0
                        else:
                            labels[ i ] = 1
                else:
                    if multiclass:
                        labels[ i ] = classes.index(symbols[ i ])
                    else:
                        if symbols[ i ] == 'N':
                            labels[ i ] = 0
                        else:
                            labels[ i ] = 1
        return labels

    def one_hot_labels( self, labels, symbols, aami, classes, multiclass=True ):
        for i in range(len(symbols)):
            # exclude 15 unclassifiable beats
            if symbols[ i ] in symbol2class.keys():
                if aami:
                    classe = symbol2class[ symbols[ i ] ]
                    if multiclass:
                        label = classes.index(classe)
                        labels[ i ][ label ] = 1
                    else:
                        if classe == 'N':
                            labels[ i ][ 0 ] = 0
                        else:
                            labels[ i ][ 0 ] = 1
                else:
                    if multiclass:
                        label = classes.index[ symbols[ i ] ]
                        labels[ i ][ label ] = 1
                    else:
                        if symbols[ i ] == 'N':
                            labels[ i ][ 0 ] = 0
                        else:
                            labels[ i ][ 0 ] = 1
        return labels

    def subsample_data( self, X, Y, classes, label, factor, one_hot ):
        if one_hot:
            label_data = list(filter(lambda x: np.argmax(x[ 1 ]) == classes.index(label), zip(X, Y)))
            other_data = list(filter(lambda x: np.argmax(x[ 1 ]) != classes.index(label), zip(X, Y)))
        else:
            label_data = list(filter(lambda x: x[ 1 ] == classes.index(label), zip(X, Y)))
            other_data = list(filter(lambda x: x[ 1 ] != classes.index(label), zip(X, Y)))
        label_X, label_Y = zip(*label_data)
        X, Y = zip(*other_data)
        sample_len = int(len(label_X) / factor)
        label_X = random.sample(label_X, sample_len)
        label_Y = label_Y[ :sample_len ]
        X = np.concatenate((X, label_X))
        Y = np.concatenate((Y, label_Y))
        return X, Y

    def augment_data( self, X, Y, classes, label, factor, one_hot=True ):
        if one_hot:
            label_data = list(filter(lambda x: np.argmax(x[ 1 ]) == classes.index(label), zip(X, Y)))
        else:
            classe = classes.index(label)
            print(classe)
            label_data = list(filter(lambda x: x[ 1 ] == classe, zip(X, Y)))
        label_X, label_Y = zip(*label_data)
        for i in range(factor - 1):
            X = np.concatenate((X, label_X))
            # X = np.concatenate((X, list(map(lambda x: x + random.random(), label_X))))
            Y = np.concatenate((Y, label_Y))
        return X, Y

    def filter( self, channel ):
        '''fs = 360
        # cutoff low frequency to get rid of baseline wonder
        f1 = 5
        # cutoff frequency to discard high frequency noise
        f2 = 15
        Wn = np.divide([f1 * 2, f2 * 2], fs)
        N = 3
        # bandpass
        [a, b] = signal.butter(N, Wn, 'band')
        # filtering
        filtered = signal.filtfilt(a, b, channel)'''
        baseline = medfilt(channel, 71)
        baseline = medfilt(baseline, 215)
        return np.subtract(channel, baseline)

    def read_image( self, train_size, classes=None, one_hot=True, aami=True ):
        if classes is None:
            classes = [ 'N', 'S', 'V', 'F' ]
        X_train = list()
        X_val = list()
        X_test = list()
        Y_train = list()
        Y_val = list()
        Y_test = list()
        names = list(
            {'100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
             '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
             '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231',
             '232', '233', '234'} - {'102', '104', '107', '217'})
        for name in names:
            print(name)
            record = wfdb.rdrecord(ecg_path + name, channels=[ 0, 1 ])
            record = np.transpose(record.p_signal)
            # for i in range(len(record)):
            # record[i] = self.remove_baseline(record[i])
            peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
            peaks, symbols = self.exclude_out_of_range(peaks, symbols)
            labels = self.extract_labels(aami=aami, classes=classes, one_hot=one_hot, symbols=symbols)
            beats = self.image_beats(peaks, record)
            self.train_val_test_split(X_test, X_train, X_val, Y_test, Y_train, Y_val, beats, labels, peaks, train_size)
        print(len(X_train))
        print(len(X_val))
        print(len(X_test))
        return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), np.array(X_test), np.array(
            Y_test)

    def image_beats( self, peaks, record ):
        beats_shape = (len(peaks), 2, 170, 1)
        beats = np.zeros(beats_shape)
        for index, p in enumerate(peaks):
            beat = np.zeros(beats_shape[ 1: ])
            for i in range(len(record)):
                channel = record[ i ]
                beat_data = channel[ p - 70:p + 100 ]
                beat[ i ] = np.reshape(beat_data, (beat.shape[ 1 ], beat.shape[ 2 ]))
            beats[ index ] = beat
        return beats

    def standardize( self, X_train, X_val, X_test ):
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test

    def compute_shape( self, dataset_names ):
        count = 0
        for name in dataset_names:
            peaks, symbols = ut.remove_non_beat(ecg_path + name, rule_based=False)
            peaks, symbol = self.exclude_out_of_range(peaks, symbols)
            count += len(peaks)
        return count

    def vertical_split(self, train_size, timesteps, channels=None, standardize=True, classes=None, aami=True,
                          model=None,
                          filtered=False, one_hot=True, multiclass=True, window=None, left_window=70, right_window=100 ):
        if channels is None:
            channels = [ 0 ]
        if classes is None:
            classes = [ 'N', 'S', 'V', 'F' ]
        names = list(
            {'100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
             '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207',
             '208',
             '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231',
             '232', '233', '234'} - {'102', '104', '107', '217'})
        X_train = list()
        Y_train = list()
        X_val = list()
        Y_val = list()
        X_test = list()
        Y_test = list()
        for name in names:
            beats, labels, peaks = self.extract_labeled_beats(aami=aami, classes=classes, name=name,
                                                              one_hot=one_hot, channels=channels, model=model,
                                                              filtered=filtered, multiclass=multiclass, window=window,
                                                              left_window=left_window, right_window=right_window)
            self.train_val_test_split(X_test, X_train, X_val, Y_test, Y_train, Y_val,
                                      beats, labels, peaks, train_size)
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)
        Y_test = np.array(Y_test)
        if standardize:
            X_train, X_val, X_test = self.standardize(X_train, X_val, X_test)
        if timesteps is not None:
            X_train, Y_train = self.compute_timesteps(X_train, Y_train, timesteps)
            X_val, Y_val = self.compute_timesteps(X_val, Y_val, timesteps)
            X_test, Y_test = self.compute_timesteps(X_test, Y_test, timesteps)
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def horizontal_split( self, classes=None, timesteps=None, window=None, left_window=70, right_window=100, train_db='mitdb',
                          multiclass=True, aami=True, one_hot=True, model=None, standardize=True, channels=None):
        if channels == None:
            channels = [0]
        if train_db == 'mitdb':
            train_dataset = [ '106', '112', '122', '201', '223', '230', "108", "109", "115", "116", "118", "119", "124",
                              "205", "207", "208", "209", "215", '101', '114', '203', '220' ]
        else:
            train_dataset = wfdb.get_record_list('incartdb')
        test_dataset = [ "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213",
                         "214", "219",
                         "221", "222", "228", "231", "232", "233", "234" ]
        X_train = list()
        Y_train = list()
        X_test = list()
        Y_test = list()
        for name in train_dataset:
            beats, labels, peaks = self.extract_labeled_beats(name=name, aami=aami, classes=classes, one_hot=one_hot,
                                                              model=model, train_db=train_db, multiclass=multiclass,
                                                              window=window,
                                                              left_window=left_window, right_window=right_window,
                                                              channels=channels)
            X_train.extend(beats)
            Y_train.extend(labels)
        for name in test_dataset:
            beats, labels, peaks = self.extract_labeled_beats(name=name, aami=aami, classes=classes, one_hot=one_hot,
                                                              model=model, train_db=train_db,
                                                              multiclass=multiclass, window=window, left_window=left_window,
                                                              right_window=right_window)
            X_test.extend(beats)
            Y_train.extend(labels)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.1)
        if standardize:
            X_train, X_val, X_test = self.standardize(X_train, X_val, X_test)
        if timesteps is not None:
            X_train, Y_train = self.compute_timesteps(X_train, Y_train, timesteps)
            X_val, Y_val = self.compute_timesteps(X_val, Y_val, timesteps)
            X_test, Y_test = self.compute_timesteps(X_test, Y_test, timesteps)
        return X_train, X_test, Y_train, Y_test

    def train_val_test_split( self, X_test, X_train, X_val, Y_test, Y_train, Y_val, beats, labels, peaks, train_size ):
        val_size = 0.1
        train_index = int(len(peaks) * train_size)
        val_index = int(len(peaks) * val_size)
        val_index = train_index + val_index
        X_train.extend(beats[ :train_index ])
        Y_train.extend(labels[ :train_index ])
        X_val.extend(beats[ train_index: val_index ])
        Y_val.extend(labels[ train_index:val_index ])
        X_test.extend(beats[ val_index: ])
        Y_test.extend(labels[ val_index: ])
