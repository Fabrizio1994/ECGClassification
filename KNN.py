class KNN:

    def compute_average_beat(self, prediction):
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


    def clean_prediction(self, prediction):
        average_beat = 250
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
        return self.get_index(cleaned_predictions)

    def get_index(self, cleaned_prediction):
        indexes = []
        index = 520000
        for pred in cleaned_prediction:
            if pred == 1:
                indexes.append(index)
            index += 1
        print(len(indexes))
        return indexes