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
    WINDOW_SIZES = [10]
    ANNOTATION_TYPES = ['beat']
    FEATURE_TYPES = ['sliding']

    def run_knn(self):
        for name in os.listdir("sample"):
            if name.endswith('.atr'):
                signal_name = name.replace(".atr", "")
                for ann_type in self.ANNOTATION_TYPES:
                    annotation = wfdb.rdann('annotations/' + ann_type +
                                            '/' + signal_name, 'atr')
                    locations = list(filter(lambda x:
                                            x > TEST_INDEX,
                                            annotation.sample))
                    for size in self.WINDOW_SIZES:
                        for feat_type in self.FEATURE_TYPES:
                            train_features, train_labels = fe.extract_features(
                                "sample/" + signal_name, ann_type, size,
                                features_type=feat_type)
                            knn_output = gs.grid_search(train_features,
                                                        train_labels)
                            starting_index = int(len(train_labels) / 5 * 4)
                            #cleaned = self.__clean_prediction(knn_output)
                            if feat_type == "sliding":
                                prediction = self.__get_sliding_indexes(
                                    knn_output, size)

                                test_labels = self.__get_sliding_indexes(
                                    train_labels[starting_index:], size)
                            else:
                                prediction = self.__get_indexes(knn_output)
                                test_labels = self.__get_indexes(
                                    train_labels[starting_index:])
                            eval.evaluate_prediction(prediction,
                                                     test_labels,
                                                     name,
                                                     SIG_LEN_LAST_20,
                                                     locations, size,
                                                     ann_type,
                                                     feat_type,
                                                     classifier="KNN")

    def __compute_average_beat(self, prediction):
        delta = 0
        sum = 0
        count = 0
        for label in prediction:
            if label == -1:
                delta += 1
            else:
                sum += delta
                delta = 0
                count += 1
        return sum / count

    def __clean_prediction(self, prediction):
        average_beat = self.__compute_average_beat(prediction)
        cleaned_predictions = []
        delta = 0
        inside = False
        for pred in prediction:
            if pred == -1:
                if inside:
                    delta = 0
                    inside = False
                delta += 1
                cleaned_predictions.append(-1)
            else:
                if delta >= average_beat:
                    inside = True
                    cleaned_predictions.append(1)
                else:
                    cleaned_predictions.append(-1)
                    delta += 1
        return cleaned_predictions

    def __get_indexes(self, cleaned_prediction):
        indexes = []
        index = TEST_INDEX
        for pred in cleaned_prediction:
            if pred == 1:
                indexes.append(index)
            index += 1
        print(len(indexes))
        return indexes

    def __get_sliding_indexes(self, predictions, window_size):
        indexes = []
        index = TEST_INDEX
        for pred in predictions:
            if pred == 1:
                indexes.extend([i for i in range(index, index + window_size)])
            index += window_size
        return indexes
