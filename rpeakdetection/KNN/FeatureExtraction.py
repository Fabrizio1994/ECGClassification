import wfdb
from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
import pickle

class FeatureExtraction:

    def extract_features(self,name, path, rpeak_locations, features_comb, window_size=None):
        if 'KNN_w' in features_comb:
            window_size = window_size
        else:
            window_size = 1
        if '1' in features_comb:
            channels = [0]
        elif '2' in features_comb:
            channels = [1]
        else:
            channels = [0, 1]
        filtered = 'FS' in features_comb
        print("Extracting features for signal " + path + "...")
        record = self.preprocess(channels, name, path)
        features_path = 'rpeakdetection/KNN/features/'
        if filtered:
            filtered_record = list()
            for i in range(len(channels)):
                filtered_channel = self.filter(record[i], features_comb)
                filtered_record.append(filtered_channel)
            record = filtered_record
        comb_name = features_comb[0] + '_' + features_comb[1] + '_' + features_comb[2]
        feat_name = features_path + name + '_' + comb_name + '.pkl'
        labels_name = features_path + name + '_' + comb_name + '_labels.pkl'
        #if write:
        features, labels = self.compute_features(record, rpeak_locations, comb=features_comb,window_size=window_size)
            #with open(feat_name, 'wb') as fid:
            #    pickle.dump(features, fid)
            #with open(labels_name, 'wb') as fid:
            #    pickle.dump(labels, fid)
        '''else:
            features = pickle.load(open(feat_name, 'rb'))
            labels = pickle.load(open(labels_name, 'rb'))'''
        return record, features, labels

    def preprocess(self, channels, name, path):
        if name == '114':
            if channels == [[0]]:
                record = wfdb.rdrecord(path, channels=[1])
            elif channels == [[1]]:
                record = wfdb.rdrecord(path, channels=[0])
            else:
                record = wfdb.rdrecord(path, channels=channels)
        else:
            record = wfdb.rdrecord(path, channels=channels)
        record = np.transpose(record.p_signal)
        return record

    def compute_features(self, record, rpeak_locations,comb, window_size):
        sig_len = len(record[0])
        n_channels = len(record)
        n_regions = int(sig_len / window_size)
        # split record in regions of size window_size. If more then 1 channel, concatenates the regions
        features = np.concatenate(np.reshape(record,(n_channels,n_regions,window_size)), axis=1)
        if 'KNN_w' in comb:
            indexes = np.split(np.arange(sig_len), n_regions)
            labels = list(map(lambda x: int(len(set(x).intersection(set(rpeak_locations)))!=0), indexes))
        else:
            half_qrs_width = 18
            labels = np.zeros(sig_len, dtype=np.int8)
            for peak in rpeak_locations:
                for q in range(peak-half_qrs_width, peak+half_qrs_width):
                    # avoids underflow and overflow
                    if 0 <= q < sig_len:
                        labels[q] = 1
        return features, labels


    def filter(self, record, comb):
        fs = 360
        # cutoff low frequency to get rid of baseline wonder
        f1 = 5
        # cutoff frequency to discard high frequency noise
        f2 = 15
        Wn = np.divide([f1*2, f2*2],fs)
        N = 3
        # bandpass
        [a, b] = signal.butter(N, Wn, 'band')
        # filtering
        filtered = signal.filtfilt(a, b, record)
        if 'KNN_s' in comb:
            ecg_d = [0]
            diff = np.diff(filtered)
            ecg_d.extend(diff)
            return ecg_d
        else:
            vector = [1, 2, 0, -2, -1]
            int_c = 160 / fs
            # 5.1 since in signal 100 we must include 5
            b = interp1d(range(1, 6), [i * fs / 8 for i in vector])(np.arange(1, 5.1, int_c))
            ecg_d = signal.filtfilt(b, 1, filtered)
            # print(ecg_d[:5])
            ecg_d = ecg_d / np.max(ecg_d)
            ''' Squaring nonlinearly enhance the dominant peaks '''
            ecg_s = ecg_d ** 2
            return ecg_s



