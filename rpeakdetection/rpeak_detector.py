import os
import wfdb
import numpy as np
import scipy.signal
import scipy.ndimage
from rpeakdetection.Utility import Utility

util = Utility()

class RPeakDetector:

    def detect_beats(self,
            ecg,  # The raw ECG signal
            rate,  # Sampling rate in HZ
            # Window size in seconds to use for
            ransac_window_size=5.0,
            # Low frequency of the band pass filter
            lowfreq=5.0,
            # High frequency of the band pass filter
            highfreq=15.0,
    ):
        """
        ECG heart beat detection based on
        http://link.springer.com/article/10.1007/s13239-011-0065-3/fulltext.html
        with some tweaks (mainly robust estimation of the rectified signal
        cutoff threshold).
        """

        ransac_window_size = int(ransac_window_size * rate)

        lowpass = scipy.signal.butter(1, highfreq / (rate / 2.0), 'low')
        highpass = scipy.signal.butter(1, lowfreq / (rate / 2.0), 'high')

        # x is the array of data to be filtered.
        ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
        ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)

        # Square (=signal power) of the first difference of the signal
        # This is done to obtain a positive-valued signal
        diff_ecg = np.diff(ecg_band)
        diff_ecg_powered = diff_ecg ** 2

        # Robust threshold and normalizator estimation
        # The thresholding process eliminates spurious noise spikes and tends to reduce the number of FP detections
        thresholds = []
        max_powers = []
        for i in range(int(len(diff_ecg_powered) / ransac_window_size)):
            # slice is used to slice a given sequence. Slice represents the indices specified by range(start, stop, step)
            sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
            d = diff_ecg_powered[sample]
            thresholds.append(0.5 * np.std(d))
            max_powers.append(np.max(d))

        threshold = np.median(thresholds)
        max_power = np.median(max_powers)
        diff_ecg_powered[diff_ecg_powered < threshold] = 0

        diff_ecg_powered /= max_power
        diff_ecg_powered[diff_ecg_powered > 1.0] = 1.0
        square_decg_power = diff_ecg_powered ** 2

        # Shannon energy transformation improves the detection accuracy under ECG signal with smaller and wider QRS complexes
        # To compute Shannon energy, the thresholded energy signal is first normalized.

        shannon_energy = -square_decg_power * np.log(square_decg_power)
        shannon_energy[~np.isfinite(shannon_energy)] = 0.0

        mean_window_len = int(rate * 0.125 + 1)
        lp_energy = np.convolve(shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode='same')
        # lp_energy = scipy.signal.filtfilt(*lowpass2, x=shannon_energy)

        lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 8.0)
        lp_energy_diff = np.diff(lp_energy)

        zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
        zero_crossings = np.flatnonzero(zero_crossings)
        zero_crossings -= 1
        return zero_crossings

    def write_output(self, peaks_folder, name, prediction):
        file = open(peaks_folder + name, "w")
        for pred in prediction:
            file.write("%s\n" % str(pred))

    def get_rpeaks(self, ecg, zero_crossings, qrs_width):
        prediction =[]
        for zero_crossing in zero_crossings:
            peak_index = self.peak(ecg, zero_crossing, qrs_width)
            prediction.append(peak_index)
        return prediction

    def peak(self, channel, zero_crossing, qrs_width):
        window_size = qrs_width / 2
        # overflow
        if int(zero_crossing + window_size) >= len(channel):
            indexes = range(int(zero_crossing - window_size), len(channel))
        else:
            indexes = range(int(zero_crossing - window_size), int(zero_crossing + window_size + 1))
        max = abs(channel[zero_crossing])
        rpeak = zero_crossing
        for index in indexes:
            if abs(channel[index]) > max:
                max = channel[index]
                rpeak = index
        return rpeak

    def evaluate(self, rpeaks, name, evaluation_width, rule_based, test_index=None):
        real_locations = util.remove_non_beat(name, rule_based)[0]
        if test_index is not None:
            real_locations = list(filter(lambda x: x >= test_index, real_locations))
        window_size = int(evaluation_width / 2)
        Y = list()
        for y in real_locations:
            Y.extend([y + q for q in range(-window_size, window_size)])
        recall = len(set(rpeaks).intersection(set(Y))) / len(real_locations)
        if len(rpeaks) != 0:
            precision = len(set(rpeaks).intersection(set(Y))) / len(rpeaks)
        else:
            precision = 0
        #print("recall")
        #print(recall)
        #print("precision")
        #print(precision)
        return recall, precision



if __name__ == '__main__':
    rpd = RPeakDetector()
    db = 'mitdb'
    peaks_folder = "../data/peaks/rpeak_detector/" + db + '/'
    ecg_folder = "../data/ecg/" + db + "/"
    channel_number = 0
    sampling_rate = 360
    names_path = "../data/mitdb_names.txt"
    qrs_width = 32
    evaluation_width = 32
    # os.mkdir("../data/peaks")
    # os.mkdir("../data/peaks/rpeak_detector")
    # os.mkdir(peaks_folder)
    names = open(names_path, 'r')
    recalls = list()
    precisions = list()
    for name in names:
        name = name.replace("\n", "")
        record = wfdb.rdrecord(ecg_folder + name, channels=[channel_number])
        ecg = record.p_signal.flatten()
        zero_crossings = rpd.detect_beats(ecg, sampling_rate)
        rpeaks = rpd.get_rpeaks(ecg, zero_crossings, qrs_width)
        rpd.write_output(peaks_folder, name, rpeaks)
        recall, precision = rpd.evaluate(rpeaks, os.path.join(ecg_folder, name), evaluation_width)
        recalls.append(recall)
        precisions.append(precision)
    print("average precision")
    print(np.mean(np.array(precisions)))
    print("average recall")
    print(np.mean(np.array(recalls)))


