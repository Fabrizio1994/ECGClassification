import wfdb
import os
from GridSearch import GridSearch
from FeatureExtraction import FeatureExtraction
from Evaluation import Evaluation

SIG_LEN = 650000
SIG_LEN_LAST_20 = int(SIG_LEN/5)
TEST_INDEX = SIG_LEN - SIG_LEN_LAST_20

eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    WINDOW_SIZES = [10, 20, 50]
    ANNOTATION_TYPES = ['cleaned']
    FEATURE_TYPES = ['sliding']
    #FEATURE_TYPES = ['fixed', 'sliding']

    def run_knn(self):
        for name in os.listdir("sample"):
            if name.endswith('.atr'):
                signal_name = name.replace(".atr", "")
                for ann_type in self.ANNOTATION_TYPES:
                    for size in self.WINDOW_SIZES:
                        for feat_type in self.FEATURE_TYPES:
                            train_features, train_labels = fe.extract_features("sample/" + signal_name, ann_type, size,
                                                                               features_type=feat_type)
                            tn, fp, fn, tp = gs.grid_search(train_features, train_labels)
                            eval.write_knn_prediction(tn, fp, fn, tp, signal_name, size, ann_type, 'KNN', feat_type)