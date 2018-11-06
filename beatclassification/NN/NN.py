from keras.models import Sequential
from keras.layers import Dense, Flatten, ConvLSTM2D, BatchNormalization, Conv2D, MaxPool2D
from collections import defaultdict
from beatclassification.Preprocessing import Preprocessing
from keras.initializers import glorot_normal, zeros
from keras.optimizers import Adam
from keras.activations import elu
from keras import callbacks
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import scikitplot as splt
import os
import random
from beatclassification.data_visualization import data_visualization
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        plt.close()
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        plt.plot(self.val_f1s, label='f1')
        plt.plot(self.val_precisions, label='precision')
        plt.plot(self.val_recalls, label='recall')
        plt.legend()
        plt.show()
        return


metrics = Metrics()
dv = data_visualization()
# os.system('rm -r -d beatclassification/NN/Graph')
# tbCallBack = callbacks.TensorBoard(log_dir='beatclassification/NN/Graph', histogram_freq=1, write_graph=True,
# write_images=True)
prep = Preprocessing()
symbol2class = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
                "A": "S", "a": "S", "J": "S", "S": "S",
                "V": "V", "E": "V",
                "F": "F"}
                #'/': 'Q', 'f': 'Q', 'Q': 'Q'}
aami_classes = ['N', 'S', 'V', 'F']
classes = ['N', 'L', 'R', 'V', 'A', 'F', 'E', 'a']
train_dataset = ['106', '112', '122', '201', '223', '230', "108", "109", "115", "116", "118", "119", "124",
                 "205", "207", "208", "209", "215", '101', '114', '203', '220']
test_dataset = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219",
                "221", "222", "228", "231", "232", "233", "234"]
# normal nn
train_shape = (51011, 170)
test_shape = (49701, 170)
#CONV
train_im_shape = (51011, 2, 170, 1)
test_im_shape = (499701, 2, 170, 1)
aami = True
distribution = dv.data_distribution(train_dataset, aami=aami)
dv.data_distribution(test_dataset, aami=aami)
X_train , Y_train = prep.read_image(train_dataset, train_im_shape, filtered=False)
X_test, Y_test = prep.read_image(test_dataset, test_im_shape, filtered=False)
#X_train, Y_train = prep.preprocess(train_dataset, train_shape)
#X_test, Y_test = prep.preprocess(test_dataset, test_shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
reduction_factor = int(distribution['N'] / distribution['V'])
X_train, Y_train = prep.subsample_data(X_train, Y_train, aami_classes, 'N', reduction_factor)
for label in ['S', 'F']:
    augment_factor = int(distribution['V'] / distribution[label])
    print(augment_factor)
    X_train, Y_train = prep.augment_data(X_train, Y_train, aami_classes, label, augment_factor)
print('train distribution')
print(dv.distribution(Y_train, aami_classes))
print('val distribution')
print(dv.distribution(Y_val, aami_classes))
print('test distribution')
print(dv.distribution(Y_test, aami_classes))

model = Sequential()
#model.add(Dense(32, input_shape=(170,), activation='relu'))
model.add(Conv2D(128, (2,10), input_shape=train_im_shape[1:], activation='relu'))
model.add(MaxPool2D(1))
#model.add(Conv2D(64, 1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(aami_classes), activation='softmax'))
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)
model.fit(X_train, Y_train, epochs=40, batch_size=32, validation_data=(X_val, Y_val), callbacks=[metrics])
predictions = model.predict(X_test, batch_size=32)
predictions = list(map(lambda x: np.argmax(x), predictions))
Y_test = list(map(lambda x: np.argmax(x), Y_test))
splt.metrics.plot_confusion_matrix(Y_test, predictions)
print("per class precision")
precision = precision_score(Y_test, predictions, average=None)
print(precision)
print("average precision")
print(np.mean(precision))
print('per class recall')
recall = recall_score(Y_test, predictions, average=None)
print(recall)
print('average recall')
print(np.mean(recall))
plt.show()
plt.close()

if not aami:
    # aami evaluation
    Y_test = list(map(lambda x: aami_classes.index(symbol2class[classes[x]]), Y_test))
    predictions = list(map(lambda x: aami_classes.index(symbol2class[classes[x]]), predictions))
    splt.metrics.plot_confusion_matrix(Y_test, predictions)
    print("per class precision")
    precision = precision_score(Y_test, predictions, average=None)
    print(precision)
    print("average precision")
    print(np.mean(precision))
    print('per class recall')
    recall = recall_score(Y_test, predictions, average=None)
    print('average recall')
    print(np.mean(recall))
    plt.show()
