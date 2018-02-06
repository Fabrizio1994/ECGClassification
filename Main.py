from FeatureExtraction import FeatureExtraction
from sklearn.neighbors import KNeighborsClassifier
from GridSearch import GridSearch

fe = FeatureExtraction()

features, labels = fe.extract_features('100')
knn = KNeighborsClassifier()
knn_parameters = {'n_neighbors': [1, 3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'p': [1, 2],
                  }

target_names = ['peak', 'not-peak']
gs = GridSearch(knn, knn_parameters, features, labels, target_names)

