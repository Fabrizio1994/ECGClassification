import wfdb
import os
from GridSearch import GridSearch
from FeatureExtraction import FeatureExtraction
from Evaluation import Evaluation
from collections import defaultdict

SIG_LEN = 650000
SIG_LEN_LAST_20 = int(SIG_LEN/5)
TEST_INDEX = SIG_LEN - SIG_LEN_LAST_20

eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    WINDOW_SIZES = [10, 20, 50]
    ANNOTATION_TYPES = ['beat']
    FEATURE_TYPES = ['sliding']
    #FEATURE_TYPES = ['fixed', 'sliding', 'on_annotation]

    def run_knn(self):
        results = defaultdict(list)
        for name in os.listdir("sample"):
            if name.endswith('.atr'):
                signal_name = name.replace(".atr", "")
                for ann_type in self.ANNOTATION_TYPES:
                    for size in self.WINDOW_SIZES:
                        for feat_type in self.FEATURE_TYPES:
                            train_features, train_labels = fe.extract_features("sample/" + signal_name, ann_type, size,
                                                                               features_type=feat_type, channels_ids=[0]
                                                                               )
                            tn, fp, fn, tp = gs.grid_search(train_features, train_labels)
                            se = eval.compute_sensitivity(tn, fp, fn, tp, signal_name, size, ann_type, feat_type)
                            results[signal_name].append(se)
        eval.write_knn_prediction(results)