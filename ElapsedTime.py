import time
from sklearn.neighbors import KNeighborsClassifier
from FeatureExtraction import FeatureExtraction
import numpy as np

fe = FeatureExtraction()


def knn_prediction(sample_name):
    KNNfeatures = np.asarray(fe.extract_features(sample_name))
    labels = np.asarray(fe.define_2class_labels(sample_name))
    knn = KNeighborsClassifier(n_neighbors=9, weights='uniform', p=2)
    knn.fit(KNNfeatures, labels)
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
                tp +=1
            else:
                fp += 1
        elif y == labels[i]:
            tn += 1
        else:
            fn += 1
        i += 1

    print("#peaks: " + str(peak_count))
    print("elapsed time: " + str(elapsed))
    print("FP = "+str(fp))
    print("FN = "+str(fn))
    print("TP = "+str(tp))
    print("TN = "+ str(tn))

if __name__ == '__main__':
    knn_prediction('100')




