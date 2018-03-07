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
        average_beat = 300
        result = []
        delta = 0
        inside = False
        for pred in prediction:
            if pred == -1:
                if inside:
                    delta = 0
                    inside = False
                delta += 1
                result.append(-1)
            else:
                if delta >= average_beat:
                    inside = True
                    result.append(1)
                else:
                    result.append(-1)
                    delta += 1
        return result



    def evaluate_results(self, predict, Ytest):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        j = 0
        for pred in predict:
            if pred == 1:
                if pred == Ytest[j]:
                    TP += 1
                else:
                    FP += 1
            else:
                if pred == Ytest[j]:
                    TN += 1
                else:
                    FN += 1
            j += 1
        file = open("report_grad.tsv", "a")
        file.write("TP:%s\tTN:%s\tFP:%s\tFN:%s\n" % (str(TP), str(TN), str(FP), str(FN)))



    def get_index(self, cleaned_prediction):
        indexes = []
        for pred in cleaned_prediction:
            for j in range(len(cleaned_prediction)):
                if pred == 1:
                    indexes.append(j)
        return indexes