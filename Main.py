import wfdb
import os
from IPython.display import display
import pprint as pp

# wfdb.dldatabase('mitdb', os.getcwd())

#To plot a single record
#record = wfdb.rdsamp('232')
#wfdb.plotrec(record, title='Record 232 from MITDB')
#display(record.__dict__)

#signals, fields = wfdb.srdsamp('232', sampfrom=100, sampto=2000)

#display(signals)
#display(fields)

#annotation = wfdb.rdann('232', 'atr', sampfrom=100, sampto=20000)
#annotation.fs = 360
#wfdb.plotann(annotation, timeunits='minutes')

signal_length = 650000
interval = 2000
names = []
record_name = {}

for signal_name in os.listdir(os.getcwd()+'/samples'):
    signal = signal_name.split('.')
    start = 0
    end = interval
    if signal[0] not in names:
        names.append(signal[0])
        count = 0
        while end <= signal_length:
            record = wfdb.rdsamp('samples/' + signal[0], sampfrom=start, sampto=end)
            start = end
            end = end + interval
            record_name[signal[0]+"_"+str(count)] = record
            count += 1


