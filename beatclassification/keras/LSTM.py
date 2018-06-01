from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from pandas_ml import ConfusionMatrix
import numpy as np

batch_size = 128
window_size = 3
data_dimension = 680


class LSTM_NN:

    def predict(self, X_train, Y_train, X_test, Y_test, weights):
        model = Sequential()
        model.add(LSTM(units=64, input_shape=(window_size, data_dimension)))
        model.add(Dense(units=64))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=4, activation='softmax'))
        model.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')
        i = 0
        while i < X_train.shape[0] - batch_size:
            iter_number = int(i / batch_size)
            total_number = int(X_train.shape[0] / batch_size)
            print(str(iter_number) + " of " + str(total_number))
            x_batch = X_train[i:i + batch_size]
            y_batch = Y_train[i:i + batch_size]
            # model.train_on_batch(x_batch, y_batch, class_weight=labels_counter)
            model.train_on_batch(x_batch, y_batch, class_weight=weights)
            i += batch_size
        pred = model.predict_classes(X_test, batch_size=128)
        y_true = Y_test
        # from one-hot to standard representation
        y_true = list(map(lambda x : np.argmax(x), y_true))
        cm = ConfusionMatrix(y_true, pred)
        print(cm.stats())
        return pred
