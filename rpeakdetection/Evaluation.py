from rpeakdetection.Utility import Utility

util = Utility()


class Evaluation:

    def evaluate(self, rpeaks, name, evaluation_width, rule_based, test_index=None):
        real_locations = util.remove_non_beat(name, rule_based)[0]
        if test_index is not None:
            real_locations = list(filter(lambda x: x >= test_index, real_locations))
        window_size = int(evaluation_width / 2)
        Y = list()
        for y in real_locations:
            Y.extend([y + q for q in range(-window_size, window_size)])
        filtered_peaks = list()
        prev = 0
        for peak in rpeaks:
            if peak - prev > evaluation_width:
                filtered_peaks.append(peak)
                prev = peak
        correct_detected = set(filtered_peaks).intersection(set(Y))
        recall = len(correct_detected) / len(real_locations)
        if len(rpeaks) != 0:
            precision = len(correct_detected) / len(rpeaks)
        else:
            precision = 0
        return recall, precision
