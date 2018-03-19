from KNN import KNN
import time

knn = KNN()
start_time = time.time()
knn.run_knn()
end_time = time.time()

print('Time needed to execute KNN with window = [10, 20, 50], ann_type = cleaned, feat_type = [fixed, on_ann, sliding] '
      'equal to = ' + str(end_time-start_time))
