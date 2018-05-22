from beatclassification.FeatureExtraction import FeatureExtraction
from beatclassification.SVCGridSearch import SVCGridSearch
from beatclassification.MulticlassEvaluation import MulticlassEvaluation
from beatclassification.SoftmaxRegression import SoftmaxRegression
fe = FeatureExtraction()
me = MulticlassEvaluation()
supraventricular_db = [i for i in range(800,813)]
train_dataset = ["101", "106", "108","109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203",
                 "205", "207", "208", "209", "215", "220", "223", "230"]
train_dataset.extend([str(e) for e in supraventricular_db])
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213","214", "219",
                "221", "222", "228", "231", "232", "233", "234"]

x_train, y_train = fe.extract(train_dataset)
x_test, y_test = fe.extract(test_dataset)
# fit method
svc = SVCGridSearch(x_train, y_train, x_test)
se = me.evaluate(y_test, svc.y_predicted)
#softmax = SoftmaxRegression(x_train, y_train, x_test, y_test)