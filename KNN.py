from sklearn.neighbors import KNeighborsClassifier
class KNN:
    def __init__(self,Xtrain,Ytrain, Xtest, Ytest):
        KNN = KNeighborsClassifier(n_neighbors=5)
        KNN.fit(Xtrain, Ytrain)
        predict = KNN.predict(Xtest)
        self.evaluate_results(predict, Ytest)

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
        print("TP:"+str(TP)+" TN:"+str(TN)+" FP:" + str(FP)+" FN:" + str(FN))

