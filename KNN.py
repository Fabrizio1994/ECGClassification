from sklearn.neighbors import KNeighborsClassifier
class KNN:
    def __init__(self):
        self.classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='distance')

    def evaluate_results(self, predict, Ytest):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        j = 0
        for pred in predict:
            if pred == 1:
                if pred == Ytest[j]:
                    TP +=1
                else:
                    FP +=1
            else:
                if pred == Ytest[j]:
                    TN += 1
                else:
                    FN += 1
            j += 1
        file = open("report.tsv", "a")
        file.write("TP:%s\tTN:%s\tFP:%s\tFN:%s\n"%(str(TP),str(TN),str(FP),str(FN)))

    def clean_prediction(self, predict):
        average_beat = self.compute_average_beat(predict)
        result = []
        delta = 0
        for pred in predict:
            if pred == -1:
                delta += 1
                result.append(-1)
            else:
                if delta >= average_beat:
                    result.append(1)
                    delta = 0
                else:
                    result.append(-1)
                    delta += 1
        return result

    def compute_average_beat(self, predict):
        delta = 0
        sum = 0
        count = 0
        for label in predict:
            if label == -1:
                delta += 1
            else:
                sum += delta
                delta = 0
                count += 1
        return sum / count

