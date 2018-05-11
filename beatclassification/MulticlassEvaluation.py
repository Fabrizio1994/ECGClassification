from collections import defaultdict
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix

class MulticlassEvaluation:

    def evaluate(self, Y_true, Y_predicted):
        #conf_mat = confusion_matrix(Y_true, Y_predicted)
        cm = ConfusionMatrix(Y_true, Y_predicted)
        cm.print_stats()
        '''for i in range(0, 4):
            TP = conf_mat[i, i]
            FP = sum(conf_mat[:, i]) - conf_mat[i, i]
            TN = sum(sum(conf_mat)) - sum(conf_mat[i, :]) - sum(conf_mat[:, i]) + conf_mat[i, i]
            FN = sum(conf_mat[i, :]) - conf_mat[i, i]
            print(i)
            acc = (TP + TN) / (TP +FP + FN + TN)
            print(acc)'''


