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
    # modify the method write_knn_predictions accoording to the lenght of this array
    WINDOW_SIZES = [10, 20, 30, 50, 70, 90, 110, 130, 150]
    EVALUATION_WINDOW_SIZE = 10
    # train fixed and sliding separately
    FEATURE_TYPE = 'fixed'
    DB = "incartdb"
    # train one and two channels separately
    CHANNEL_IDS = [0]

    def rpeak_detection(self):
        results = defaultdict(list)
        for name in sorted(os.listdir("sample/"+self.DB)):
            if name.endswith('.atr'):
                name = name.replace(".atr", "")
                path = ("sample/" + self.DB + "/" + name)
                for size in self.WINDOW_SIZES:
                    annotation = wfdb.rdann(path, 'atr')
                    train_features, train_labels = fe.extract_features(path, annotation, size,
                                                                       features_type=self.FEATURE_TYPE,
                                                                       channels_ids=self.CHANNEL_IDS)
                    # signal first channel. Needed for extracting peaks from predicted regions
                    signal = fe.channels_map[0]
                    test_index = int(len(signal) / 5) * 4
                    locations = list(filter(lambda x: x > test_index, annotation.sample))
                    # KNN training and peak detection from the KNN output
                    peaks = gs.rpeak_gridsearch(train_features, train_labels, locations, size, signal)
                    # contains the interval of size=evaluation_window_size around the annotations
                    Y_test = eval.get_labels(locations, self.EVALUATION_WINDOW_SIZE)
                    fn, fp, tp, tn = eval.confusion_matrix(Y_test, peaks)[0:4]
                    se = eval.compute_sensitivity(fp, fn, tp)
                    results[name].append(se)

        eval.write_knn_prediction(results)

    def qrs_detection(self):
        results = defaultdict(list)
        for name in sorted(os.listdir("sample/"+self.DB)):
            if name.endswith('.atr'):
                name = name.replace(".atr", "")
                path = ("sample/"+self.DB+"/"+name)
                for size in self.WINDOW_SIZES:
                    for feat_type in self.FEATURE_TYPES:
                        for channels in self.CHANNEL_IDS:
                            annotation = wfdb.rdann(path, 'atr')
                            train_features, train_labels = fe.extract_features(path, annotation, size,
                                                                               features_type=feat_type,
                                                                               channels_ids=channels)
                            tn, fp, fn, tp = gs.qrs_gridsearch(train_features, train_labels)
                            se = eval.compute_sensitivity(fp, fn, tp)
                            results[name].append(se)
        eval.write_knn_prediction(results)




