import wfdb
import numpy as np
from rpeakdetection.Utility import Utility
import matplotlib.pyplot as plt


ut = Utility()
ecg_path = 'data/ecg/mitdb/'
class2symbols = {'N': ['N', 'L', 'R', 'e', 'j'],
                 'S': ['A', 'a', 'J', 'S'],
                 'V': ['V', 'E'],
                 'F': ['F'],
                 'Q': ['/', 'f', 'Q']}
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F",
                '/': 'Q', 'f': 'Q', 'Q': 'Q'}


class data_visualization():

    def plot_beats(self, label):
        for name in wfdb.get_record_list('mitdb'):
            print(name)
            # noinspection PyRedeclaration
            record = wfdb.rdrecord(ecg_path + name)
            record = np.transpose(record.p_signal)
            peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
            s_pairs = list(filter(lambda x: x[1] == label, zip(peaks, symbols)))
            s_beats_first = [record[0][pair[0] - 70:pair[0] + 100] for pair in s_pairs]
            if len(s_beats_first) > 0:
                for beat in s_beats_first:
                    plt.plot(beat)
        plt.title(label)
        plt.show()

    def plot_beats_dataset(self, X, Y, dataset_name, label_index, one_hot=True):
        if one_hot:
            Y = list(map(lambda x : np.argmax(x), Y))
        s_pairs = list(filter(lambda x: x[1] == label_index, zip(X, Y)))
        s_beats, _ = zip(*s_pairs)
        plt.plot(np.mean(s_beats, axis=0))
        plt.savefig('beatclassification/beat_images/'+dataset_name+ ' average S beat.png')
        plt.close()

    def data_distribution(self, dataset, aami):
        from collections import defaultdict
        distribution = defaultdict(int)
        for name in dataset:
            peaks, symbols = ut.remove_non_beat(ecg_path + name, False)
            for sym in symbols:
                if aami:
                    sym = symbol2class[sym]
                distribution[sym] += 1
        sorted_by_value = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        print(sorted_by_value)
        return distribution

    def distribution(self, Y,  classes, multiclass=True):
        from collections import defaultdict
        distribution = defaultdict(int)
        for one_hot in Y:
            if multiclass:
                index = np.argmax(one_hot)
            else:
                index= int(one_hot[0])
            symbol = classes[index]
            distribution[symbol] += 1
        return distribution

    def plot_wrong_predictions( self, predicted, target, beats):
        s_wrong = list(filter(lambda x: x[ 0 ] == 0 and x[ 1 ] == 1, zip(predicted, target, beats)))
        print(len(s_wrong))
        _, _, wrong_beats = zip(*s_wrong)
        wrong_beats = list(map(lambda x: x[ -1 ], wrong_beats))
        plt.close()
        for beat in wrong_beats:
            plt.plot(beat)
        plt.title('wrong classified S beats')
        plt.show()
        plt.close()



if __name__ == '__main__':
    train_dataset = ['106', '112', '122', '201', '223', '230', "108", "109", "115", "116", "118", "119", "124",
                     "205", "207", "208", "209", "215", '101', '114', '203', '220']
    test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214",
                    "219",
                    "221", "222", "228", "231", "232", "233", "234"]
    dv = data_visualization()
    dv.data_distribution(train_dataset)
    dv.data_distribution(test_dataset)
