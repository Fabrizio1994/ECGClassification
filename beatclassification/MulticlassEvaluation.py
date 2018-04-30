from collections import defaultdict
class MulticlassEvaluation:

    def evaluate(self, Y_true, Y_predicted, index):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        for j in range(len(Y_true)):
            label = Y_true[j]
            pred = Y_predicted[j]
            if label == pred:
                tp[label] += 1
            else:
                fp[pred] += 1
                fn[label] += 1
        Y_true = set(Y_true)
        Y_predicted = set(Y_predicted)
        labels = Y_true.union(Y_predicted)
        se = {}
        for lab in labels:
            if tp[lab] + fn[lab] != 0:
                se[lab] = tp[lab]/(tp[lab] + fn[lab])
            else:
                se[lab] = 0
            file = open("results.tsv","a")
            file.write("%s\t%s\n" % (index[lab], str(se[lab])))




