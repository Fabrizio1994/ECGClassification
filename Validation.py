import os
from KNN import KNN
from FeatureExtraction import FeatureExtraction
from Utility import Utility

ut = Utility()
fe = FeatureExtraction()
knn = KNN()

class Validation:
    def validate_for_all(self, train_features, train_labels):
        classifier = knn.classifier
        print("training KNN...")
        classifier.fit(train_features, train_labels)
        for sample in os.listdir("features"):
            if(sample.endswith('.tsv')):
                print("validation for "+sample)
                self.validate(classifier, sample)

    def validate(self, classifier, test_sample):
        test_features, test_labels = ut.read_signal(test_sample)
        file = open("report.tsv", "a")
        file.write("%s\n" %(test_sample))
        predicted = classifier.predict(test_features)
        knn.evaluate_results(predicted, test_labels)