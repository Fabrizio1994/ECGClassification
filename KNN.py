import wfdb
import os
from GridSearch import GridSearch
from FeatureExtraction import FeatureExtraction
from Evaluation import Evaluation
from collections import defaultdict

eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    # modify the method write_knn_predictions according to the lenght of this array
    WINDOW_SIZES = [10, 20, 30, 50, 70, 90, 110, 130, 150]
    EVALUATION_WINDOW_SIZE = 10
    DB = "mitdb"
    # train one and two channels separately
    CHANNEL_IDS = [0]

    def rpeak_detection(self):
        results = defaultdict(list)
        signal_peaks = {}
        for name in sorted(os.listdir("sample/" + self.DB)):
            if name.endswith('.atr'):
                name = name.replace(".atr", "")
                path = ("sample/"+self.DB+"/"+name)
                for size in self.WINDOW_SIZES:
                    annotation = wfdb.rdann(path, 'atr')
                    train_features, train_labels = fe.extract_features(path, annotation, size,
                                                                       channels_ids=self.CHANNEL_IDS)
                    signal = fe.channels_map[0]
                    Y_test, Y_predicted = gs.predict(train_features, train_labels, size)
                    tn, fp, fn, tp = gs.qrs_confusion_matrix(Y_test, Y_predicted)
                    se = eval.compute_sensitivity(fp, fn, tp)
                    print(se)
                    results[name].append(round(se, 3))
                    peaks = gs.get_peaks(Y_predicted, size, signal)
                    signal_peaks[name] = peaks
        eval.write_knn_prediction(results)




