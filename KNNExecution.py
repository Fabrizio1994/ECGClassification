import time
from sklearn.neighbors import KNeighborsClassifier
from FeatureExtraction import FeatureExtraction
from Utility import Utility
import numpy as np
import wfdb

fe = FeatureExtraction()
ut = Utility()


def knn_prediction(sample_name):
    KNNfeatures, labels = ut.load_feature(sample_name)
    annotations = wfdb.rdann('samples/'+sample_name,'atr')
    knn = KNeighborsClassifier(n_neighbors=11, p=2)
    knn.fit(np.asarray(KNNfeatures), np.asarray(labels))
    start_time = time.time()
    pred = knn.predict(KNNfeatures)
    elapsed = time.time() - start_time
    peak_count = 0
    i = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for y in pred:
        if y == 1:
            peak_count += 1
            if y == labels[i]:
                tp += 1
            else:
                fp += 1
        elif y == labels[i]:
            tn += 1
        else:
            fn += 1
        i += 1
    #TODO: fix actual peaks computation. Do not consider non beat annotations
    print("actual peaks :"+str(len(annotations.sample)))
    print("#peaks: " + str(peak_count))
    print("elapsed time: " + str(elapsed))
    print("FP = "+str(fp))
    print("FN = "+str(fn))
    print("TP = "+str(tp))
    print("TN = "+ str(tn))

if __name__ == '__main__':
    knn_prediction('101')




