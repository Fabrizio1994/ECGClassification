import wfdb
from IPython.display import display
from scipy import signal
from numpy import gradient

# wfdb.dldatabase('mitdb', os.getcwd())

#To plot a single record
#record = wfdb.rdsamp('232')
#wfdb.plotrec(record, title='Record 232 from MITDB')
#display(record.__dict__)



record = wfdb.rdsamp('samples/100')
annotation = wfdb.rdann('samples/100', 'atr')
peak_location = annotation.sample
labels = []

first_channel = []
second_channel = []
record.p_signals = signal.lfilter([-16, -32], [-1], record.p_signals)

for elem in record.p_signals:
    first_channel.append(elem[0])
    second_channel.append((elem[1]))
gradient_channel1 = gradient(first_channel)
gradient_channel2 = gradient(second_channel)

features = []
for i in range(650000):
    print(i)
    features.append([gradient_channel1[i], gradient_channel2[i]])
    if i in peak_location:
        labels.append(1)
    else:
        labels.append(-1)

print(len(labels)) #650000
print(labels) 
print(labels[18]) #1





