"""
Copyright (c) 2013 Jami Pekkanen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import scipy.signal
import scipy.ndimage
import time
import os
import wfdb

# zero_crossings_folder = "rpeakdetection/output/RPeakDetector/mitdb/zero_crossings"
# peaks_folder= "rpeakdetection/output/RPeakDetector/mitdb/peaks"
class RPeakDetection:
    def get_command(self, file_name, db, output_folder):
        return "python2 rpeakdetection/RPeakDetection.py 360  < data/rpeak_detector_input/" + db + "/" + file_name + " >" \
               + output_folder + "/" + file_name

    def run_rpeak(self, db, zero_crossings_folder, peaks_folder):
        for file_name in os.listdir("data/rpeak_detector_input/" + db):
            os.system(self.get_command(file_name, db, zero_crossings_folder))
            if file_name.endswith("_1.csv"):
                signame = file_name.replace("_1.csv", "")
                prediction = self.get_prediction(db, signame, zero_crossings_folder)
                self.write_output(peaks_folder, signame, prediction)

    def write_output(self, peaks_folder, signame, prediction):
        file = open(peaks_folder + "/" + signame, "w")
        for pred in prediction:
            file.write("%s\n" % str(pred))

    def prepare_input_files(self, db):
        for name in os.listdir("sample/"+db):
            if name.endswith(".atr"):
                name = name.replace(".atr", "")
                file = open("data/rpeak_detector_input/" + db + "/" + name + ".csv", "w")
                record = wfdb.rdrecord("data/sample/"+db+"/" + name)
                for elem in record.p_signal:
                    file.write("%s\n" % (str(elem[0])))
                file.close()

    def get_prediction(self, DB, signame, zero_crossings_folder):
        record = wfdb.rdrecord('data/sample/' + DB + "/" + signame)
        channel = []
        for elem in record.p_signal:
            channel.append(elem[0])
        prediction = []
        file = open(zero_crossings_folder + "/" + str(signame) + "_1.csv", "r")
        for line in file:
            zero_crossing = int(line.replace("\n", ""))
            peak_index = self.peak(channel, zero_crossing)
            prediction.append(peak_index)
        return prediction

    def peak(self, channel, zero_crossing):
        # overflow
        window_size = 50
        if int(zero_crossing + window_size / 2) >= len(channel):
            indexes = range(int(zero_crossing - window_size /2), len(channel))
        else:
            indexes = range(int(zero_crossing - window_size / 2), int(zero_crossing + window_size / 2 + 1))
        max = abs(channel[zero_crossing])
        rpeak = zero_crossing
        for index in indexes:
            if abs(channel[index]) > max:
                max = channel[index]
                rpeak = index
        return rpeak

def detect_beats(
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
        #slice is used to slice a given sequence. Slice represents the indices specified by range(start, stop, step)
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

    #Shannon energy transformation improves the detection accuracy under ECG signal with smaller and wider QRS complexes
    #To compute Shannon energy, the thresholded energy signal is first normalized.

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


if __name__ == '__main__':
    rate = float(sys.argv[1])
    ecg = np.loadtxt(sys.stdin)
    start_time = time.time()
    peaks = detect_beats(ecg, rate)
    elapsed = time.time() - start_time
    sys.stdout.write("\n".join(map(str, peaks)))
    sys.stdout.write("\n")
