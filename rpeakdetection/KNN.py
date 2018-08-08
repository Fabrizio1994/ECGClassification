import wfdb
import os
from rpeakdetection.GridSearch import GridSearch
from rpeakdetection.FeatureExtraction import FeatureExtraction
from rpeakdetection.Evaluation import Evaluation

eval = Evaluation()
fe = FeatureExtraction()
gs = GridSearch()


class KNN:
    WINDOW_SIZES = [20]
    EVALUATION_WINDOW_SIZE = 50
    DB = "mitdb"
    # train one and two channels separately
    CHANNEL_IDS = [0]

    def rpeak_detection(self):
        for name in sorted(os.listdir("ecg/" + self.DB)):
            if name.endswith('.atr'):
                name = name.replace(".atr", "")
                path = ("ecg/"+self.DB+"/"+name)
                for size in self.WINDOW_SIZES:
                    annotation = wfdb.rdann(path, 'atr')
                    train_features, train_labels = fe.extract_features(path, annotation.sample, size,
                                                                       channels_ids=self.CHANNEL_IDS)
                    signal = fe.channels_map[0]
                    gs.predict(train_features, train_labels, size)
                    peaks = gs.get_peaks(gs.predicted, size, signal)
                    locations = list(filter(lambda x: x > 520000, annotation.sample))
                    intervals = eval.get_labels(locations, self.EVALUATION_WINDOW_SIZE)
                    eval.evaluate_rpeak_prediction(peaks, intervals, name, size, locations)
                    '''peaks_file = open(
                        "output/KNN/" + self.DB + "/rpeakdetection/two_channels/" + str(size)
                        + "/" + name + ".tsv", "a")
                    for peak in peaks:
                        peaks_file.write("%s\n" % (str(peak)))'''






