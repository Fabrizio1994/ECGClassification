import numpy as np
from scipy.interpolate import interp1d
import peakutils
from scipy import signal

class Pan:

    """

            Inputs
            ----------
             ecg : raw ecg vector signal 1d signal
             fs : sampling frequency e.g. 200Hz, 400Hz etc

            Outputs
            -------
            qrs_amp_raw : amplitude of R waves amplitudes
            qrs_i_raw : index of R waves
            delay : number of samples which the signal is delayed due to the filtering

        """
    def pan_tompkin(self, ecg, fs, filtered):

        ''' Initialize '''

        delay = 0
        skip = 0                    # Becomes one when a T wave is detected
        m_selected_RR = 0
        mean_RR = 0
        ser_back = 0

        ''' Noise Cancelation (Filtering) (5-15 Hz) '''

        if fs == 200:
            ''' Remove the mean of Signal '''
            ecg = ecg - np.mean(ecg)

            ''' Low Pass Filter H(z) = (( 1 - z^(-6))^2) / (1-z^(-1))^2 '''
            ''' It has come to my attention the original filter does not achieve 12 Hz
                b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1] 
                a = [1, -2, 1]
                ecg_l = filter(b, a, ecg)
                delay = 6
            '''
            Wn = 12*2/fs
            N = 3
            a, b = signal.butter(N, Wn, btype='lowpass')
            ecg_l = signal.filtfilt(a, b, ecg)
            ecg_l = ecg_l/np.max(np.abs(ecg_l))

            ''' High Pass Filter H(z) = (-1 + 32z^(-16) + z^(-32)) / (1+z^(-1))'''
            ''' It has come to my attention the original filter does not achieve 5 Hz
                b = np.zeros((1,33))
                b(1) = -1
                b(17) = 32
                b(33) = 1
                a = [1, 1]
                ecg_h = filter(b, a, ecg_l)  -> Without delay
                delay = delay + 16'''


            Wn = 5*2/fs
            N = 3                                           # Order of 3 less processing
            a, b = signal.butter(N, Wn, btype='highpass')             # Bandpass filtering
            ecg_h = signal.filtfilt(a, b, ecg_l)
            ecg_h = ecg_h/np.max(np.abs(ecg_h))

        else:
            ''' Band Pass Filter for noise cancelation of other sampling frequencies (Filtering)'''
            f1 = 5                                          # cutoff low frequency to get rid of baseline wander
            f2 = 15                                         # cutoff frequency to discard high frequency noise
            Wn = [f1*2/fs, f2*2/fs]                         # cutoff based on fs
            N = 3                                           # order of 3 less processing
            a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')   # Bandpass filtering
            ecg_h = signal.filtfilt(a, b, ecg)
            ecg_h = ecg_h/np.max(np.abs(ecg_h))

        ''' Derivative Filter '''
        ''' H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2)) '''

        vector = [1, 2, 0, -2, -1]
        if fs != 200:
            int_c = 160/fs
            # 5.1 since in signal 100 we must include 5
            b = interp1d(range(1, 6), [i*fs/8 for i in vector])(np.arange(1, 5.1, int_c))
        else:
            b = [i*fs/8 for i in vector]


        ecg_d = signal.filtfilt(b, 1, ecg_h)

        ecg_d = ecg_d/np.max(ecg_d)

        ''' Squaring nonlinearly enhance the dominant peaks '''

        ecg_s = ecg_d**2


        ''' Moving Average '''
        ''' Y(nT) = (1/N)[x(nT-(N-1)T) + x(nT - (N-2) T) + ... + x(nT)] '''

        temp_vector = np.ones((1, round(0.150*fs)))/round(0.150*fs)
        temp_vector = temp_vector.flatten()
        ecg_m = np.convolve(ecg_s, temp_vector)

        delay = delay + round(0.150*fs)/2


        ''' Fiducial Marks '''
        ''' Note : a minimum distance of 40 samples is considered between each R wave since in physiological
            point of view, no RR wave can occur in less than 200ms distance'''

        if filtered:
            ecg_m = ecg
        pks = []
        new_locs = []
        locs = peakutils.indexes(y=ecg_m, thres=0, min_dist=round(0.2*fs))
        for val in locs:
            new_val = val + 1
            new_locs.append(new_val)
            pks.append(ecg_m[new_val])








        ''' Initialize Some Other Parameters '''
        LLp = len(pks)

        ''' Stores QRS with respect to Signal and Filtered Signal '''
        qrs_c = np.zeros(LLp)           # Amplitude of R
        qrs_i = np.zeros(LLp)           # index
        qrs_i_raw = np.zeros(LLp)       # Amplitude of R
        qrs_amp_raw = np.zeros(LLp)     # index
        ''' Noise Buffers '''
        nois_c = np.zeros(LLp)
        nois_i = np.zeros(LLp)

        ''' Buffers for signal and noise '''

        SIGL_buf = np.zeros(LLp)
        NOISL_buf = np.zeros(LLp)
        THRS_buf = np.zeros(LLp)
        SIGL_buf1 = np.zeros(LLp)
        NOISL_buf1 = np.zeros(LLp)
        THRS_buf1 = np.zeros(LLp)

        ''' Initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE '''
        THR_SIG = np.max(ecg_m[:2*fs])*1/3                 # 0.33 of the max amplitude
        THR_NOISE = np.mean(ecg_m[:2*fs])*1/2              # 0.5 of the mean signal is considered to be noise
        SIG_LEV = THR_SIG
        NOISE_LEV = THR_NOISE


        ''' Initialize bandpath filter threshold (2 seconds of the bandpass signal) '''
        THR_SIG1 = np.max(ecg_h[:2*fs])*1/3
        THR_NOISE1 = np.mean(ecg_h[:2*fs])*1/2
        SIG_LEV1 = THR_SIG1                                 # Signal level in Bandpassed filter
        NOISE_LEV1 = THR_NOISE1                             # Noise level in Bandpassed filter



        ''' Thresholding and decision rule '''

        Beat_C = 0
        Beat_C1 = 0
        Noise_Count = 0
        locs = new_locs                     # Just to come back to the matlab variable name
        for i in range(LLp):
            ''' Locate the corresponding peak in the filtered signal '''
            if locs[i] - round(0.150*fs) >= 1 and locs[i] <= len(ecg_h):
                temp_vec = ecg_h[locs[i] - round(0.150*fs)-1:locs[i]]     # -1 since matlab works differently with indexes
                y_i = np.max(temp_vec)
                x_i = list(temp_vec).index(y_i)
            else:
                if i == 0:
                    y_i = np.max(ecg_h[1:locs[i]])
                    x_i = list(ecg_h[1:locs[i]]).index(y_i)
                    ser_back = 1
                elif locs[i] >= len(ecg_h):
                    y_i = np.max(ecg_h[locs[i] - round(0.150*fs):])
                    x_i = list(ecg_h[locs[i] - round(0.150*fs):]).index(y_i)


            ''' Update the Hearth Rate '''
            if Beat_C >= 9:
                diffRR = np.diff(qrs_i[Beat_C-8:Beat_C])            # Calculate RR interval
                mean_RR = np.mean(diffRR)                           # Calculate the mean of 8 previous R waves interval
                comp = qrs_i[Beat_C] - qrs_i[Beat_C-1]              # Latest RR
                if comp <= 0.92*mean_RR or comp >= 1.16*mean_RR:
                    ''' lower down thresholds to detect better in MVI '''
                    THR_SIG = 0.5 * THR_SIG
                    THR_SIG1 = 0.5 * THR_SIG1
                else:
                    m_selected_RR = mean_RR                         #The latest regular beats mean

            ''' Calculate the mean last 8 R waves to ensure that QRS is not '''
            if bool(m_selected_RR):
                test_m = m_selected_RR                              #if the regular RR available use it
            elif bool(mean_RR) and m_selected_RR == 0:
                test_m = mean_RR
            else:
                test_m = 0

            if bool(test_m):
                if locs[i] - qrs_i[Beat_C] >= round(1.66*test_m):     # it shows a QRS is missed
                    pks_temp = np.max(ecg_m[int(qrs_i[Beat_C] + round(0.2*fs)-1):int(locs[i]-round(0.2*fs))])  # search back and locate the max in the interval
                    locs_temp = pks_temp
                    locs_temp = qrs_i[Beat_C] + round(0.2*fs) + locs_temp - 1   # location

                    if pks_temp > THR_NOISE:
                        Beat_C = Beat_C + 1
                        qrs_c[Beat_C] = pks_temp
                        qrs_i[Beat_C] = locs_temp

                        ''' Locate in Filtered Signal '''

                        if locs_temp <= len(ecg_h):
                            y_i_t = np.max(ecg_h[int(locs_temp-round(0.150*fs)):int(locs_temp)])
                            x_i_t = list(ecg_h[int(locs_temp - round(0.150*fs)):int(locs_temp)]).index(y_i_t)
                        else:
                            y_i_t = np.max(ecg_h[int(locs_temp-round(0.150*fs)):])
                            x_i_t = list(ecg_h[int(locs_temp - round(0.150*fs)):]).index(y_i_t)

                        ''' Band Pass Signal Threshold '''
                        if y_i_t > THR_NOISE1:
                            Beat_C1 = Beat_C1 + 1
                            qrs_i_raw[Beat_C1] = locs_temp-round(0.150*fs) + x_i_t - 1   # save index of bandpass
                            qrs_amp_raw[Beat_C1] = y_i_t                                 # save amplitude of bandpass
                            SIG_LEV1 = 0.25 * y_i_t + 0.75 *SIG_LEV1                     #when found with the second threshold

                        not_nois = 1
                        SIG_LEV = 0.25 * pks_temp + 0.75 *SIG_LEV

                    else:
                        not_nois = 0


            ''' Find noise and QRS Peaks '''

            if pks[i] >= THR_SIG:
                ''' if NO QRS in 360 ms of the previous QRS See if T wave '''
                if Beat_C >= 3:
                    if locs[i] - qrs_i[Beat_C] <= round(0.36*fs):
                        Slope1 = np.mean(np.diff(ecg_m[locs[i]-round(0.075*fs):locs[i]]))          # mean slope of the waveform at that position
                        Slope2 = np.mean(np.diff(ecg_m[int(qrs_i[Beat_C] - int(round(0.075*fs))) - 1 : int(qrs_i[Beat_C])]))        # mean slope of previous R wave
                        if np.abs(Slope1) <= np.abs(0.5*Slope2):                                    # slope less then 0.5 of previous R
                            Noise_Count = Noise_Count + 1
                            nois_c[Noise_Count] = pks[i]
                            nois_i[Noise_Count] = locs[i]
                            skip = 1                                              # T wave identification
                        else:
                            skip = 0

                ''' Skip is 1 when a T wave is detected '''
                if skip == 0:
                    Beat_C = Beat_C + 1
                    qrs_c[Beat_C] = pks[i]
                    qrs_i[Beat_C] = locs[i]


                    ''' Band pass Filter check threshold '''

                    if y_i >= THR_SIG1:
                        Beat_C1 = Beat_C1 + 1
                        if bool(ser_back):
                            qrs_i_raw[Beat_C1] = x_i
                        else:
                            qrs_i_raw[Beat_C1] = locs[i] - round(0.150*fs) + x_i - 1

                        qrs_amp_raw[Beat_C1] = y_i
                        SIG_LEV1 = 0.125*y_i + 0.875*SIG_LEV1


                    SIG_LEV = 0.125*pks[i] + 0.875*SIG_LEV


            elif THR_NOISE <= pks[i] and pks[i] < THR_SIG:
                NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                NOISE_LEV = 0.125*pks[i] + 0.875 * NOISE_LEV

            elif pks[i] < THR_NOISE:
                Noise_Count = Noise_Count + 1
                nois_c[Noise_Count] = pks[i]
                nois_i[Noise_Count] = locs[i]

                NOISE_LEV1 = 0.125*y_i +0.875 *NOISE_LEV1
                NOISE_LEV = 0.125*pks[i] + 0.875*NOISE_LEV

            ''' Adjust the threshold with SNR '''

            if NOISE_LEV != 0 or SIG_LEV != 0:
                THR_SIG = NOISE_LEV + 0.25 * (np.abs(SIG_LEV - NOISE_LEV))
                THR_NOISE = 0.5 * THR_SIG

            ''' Adjust the threshold with SNR for bandpassed signal '''

            if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
                THR_SIG1 = NOISE_LEV1 + 0.25*(np.abs(SIG_LEV1 - NOISE_LEV1))
                THR_NOISE1 = 0.5* THR_SIG1


            ''' take a track of thresholds of smoothed signal '''

            SIGL_buf[i] = SIG_LEV
            NOISL_buf[i] = NOISE_LEV
            THRS_buf[i] = THR_SIG

            ''' take a track of thresholds of filtered signal '''

            SIGL_buf1[i] = SIG_LEV1
            NOISL_buf1[i] = NOISE_LEV1
            THRS_buf1[i] = THR_SIG1

            ''' reset parameters '''

            skip = 0
            not_nois = 0
            ser_back = 0



        ''' Adjust lengths '''

        qrs_i_raw = qrs_i_raw[:Beat_C1]
        qrs_amp_raw = qrs_amp_raw[:Beat_C1]
        qrs_c = qrs_c[:Beat_C]
        qrs_i = qrs_i[:Beat_C]
        qrs_i = qrs_i[1:]
        new_qrs_i = []
        for value in qrs_i:
            new_value = int(value)
            new_qrs_i.append(new_value)


        return qrs_amp_raw, new_qrs_i, delay







