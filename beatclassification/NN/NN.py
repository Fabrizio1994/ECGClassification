from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, Conv2D, Flatten, Embedding
from keras.models import Sequential
from keras import backend as K
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import wfdb
import scikitplot as splt
from time import time
from beatclassification.data_visualization import data_visualization
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score
from beatclassification.Preprocessing import Preprocessing
from beatclassification.Evaluation import Evaluation
from keras.models import load_model


dv = data_visualization()
prep = Preprocessing()
eval = Evaluation()


# noinspection PyTypeChecker
class NN:

    def create_LSTM_model( self, X_train, X_val, Y_train, Y_val, aami_classes, callbacks, n_LSTM_layers, n_dense_layers,
                           batch_size,
                           activation, timesteps, n_neurons, optimizer, batch_normalization, epochs, dropout ):
        model = Sequential()
        input_shape = (timesteps, X_train.shape[ 2 ])
        if len(aami_classes) == 2:
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
        else:
            loss = 'categorical_crossentropy'
            output_activation = 'softmax'
        if n_LSTM_layers > 1:
            model.add(
                LSTM(n_neurons, input_shape=input_shape, activation=activation, return_sequences=True))
            for i in range(n_LSTM_layers - 2):
                model.add(LSTM(n_neurons, activation=activation, return_sequences=True))
            model.add(LSTM(n_neurons, activation=activation))
        elif n_LSTM_layers == 1:
            input_shape = (timesteps, X_train.shape[ 2 ])
            model.add(
                LSTM(n_neurons, input_shape=input_shape, activation=activation))
        if n_dense_layers > 0 and n_LSTM_layers == 0:
            model.add(Dense(n_neurons, input_shape=(170,)))
            for i in range(n_dense_layers - 1):
                model.add(Dense(n_neurons, activation=activation))
                if batch_normalization:
                    model.add(BatchNormalization())
                if dropout is not None:
                    model.add(Dropout(dropout))
        elif n_dense_layers > 0:
            if batch_normalization:
                model.add(BatchNormalization())
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons, activation=activation))
                if batch_normalization:
                    model.add(BatchNormalization())
                if dropout is not None:
                    model.add(Dropout(dropout))
        model.add(Dense(Y_train.shape[-1], activation=output_activation))
        optimizer = optimizer
        model.compile(loss=loss,
                      optimizer=optimizer)
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                  callbacks=callbacks, sample_weight=None)
        return model

    def create_CNN_model( self, X_train, X_val, Y_train, Y_val, aami_classes, callbacks, n_dense_layers, batch_size,
                          activation, width, n_neurons, optimizer, batch_normalization, epochs ):
        input_shape = (2, 170, 1)
        model = Sequential()
        model.add(Conv2D(n_neurons, (2, width), input_shape=input_shape, activation=activation))
        model.add(Flatten())
        if n_dense_layers > 0:
            if batch_normalization:
                model.add(BatchNormalization())
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons, activation=activation))
                if batch_normalization:
                    model.add(BatchNormalization())
        model.add(Dense(len(aami_classes), activation='softmax'))
        optimizer = optimizer
        tensorboard = TensorBoard(log_dir="beatclassification/NN/logs/{}".format(time()), write_grads=True,
                                  histogram_freq=1)
        callbacks.append(tensorboard)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                  callbacks=callbacks)
        return model

    def beat_classification( self, classes=None, aami=True, augment=None, reduce=None, train_size=0.5, channels=None,
                             n_dense_layers=1, batch_size=32, standardize=True, multiclass=True, activation='relu',
                             timesteps=5, n_neurons=128, filtered=False, model=None, model_name='LSTM', epochs=100, train_db='mitdb',
                             dropout=0.5, vertical=True, n_LSTM_layers=1, window=None, left_window=70, right_window= 100,
                             callbacks=None, optimizer=None, batch_normalization=True, width=2, patience=10, validation=False):
        if classes is None:
            classes = ['N', 'S', 'V', 'F']
        if callbacks is None:
            callbacks = [ EarlyStopping(patience=patience, restore_best_weights=True),
                          TensorBoard(log_dir="beatclassification/NN/logs/{}".format(time()),
                                  histogram_freq=1)]
        if optimizer is None:
            optimizer = Adam()
        if channels is None:
            channels = [0]
        if vertical:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = prep.vertical_split(classes=classes,
                                                                                   timesteps=timesteps,
                                                                                   channels=channels,
                                                                                   standardize=standardize, aami=aami,
                                                                                   model=model_name, train_size=train_size,
                                                                                   multiclass=multiclass,
                                                                                   window=window,
                                                                                   left_window=left_window,
                                                                                   right_window=right_window)
        # De Chazal horizontal split
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = prep.horizontal_split(classes=classes,
                                                                                   timesteps=timesteps,
                                                                                   channels=channels,
                                                                                   standardize=standardize, aami=aami,
                                                                                   model=model_name, train_db=train_db,
                                                                                   multiclass=multiclass,
                                                                                   window=window,
                                                                                   left_window=left_window,
                                                                                   right_window=right_window)
        dv.distribution(Y_train, classes=classes, multiclass=multiclass)
        if reduce is not None:
            for label in reduce:
                reduction_factor = reduce[ label ]
                if reduction_factor > 1:
                    X_train, Y_train = prep.subsample_data(X_train, Y_train, classes, label, reduction_factor,
                                                           one_hot=True)
        if augment is not None:
            for label in augment:
                print(label)
                augment_factor = augment[ label ]
                print(augment_factor)
                if augment_factor > 1:
                    X_train, Y_train = prep.augment_data(X_train, Y_train, classes, label, augment_factor,
                                                         one_hot=True)
        X_train, Y_train = shuffle(X_train, Y_train)
        print('train distribution')
        print(dv.distribution(Y_train, classes, multiclass=multiclass))
        print('val distribution')
        print(dv.distribution(Y_val, classes, multiclass=multiclass))
        print('test distribution')
        print(dv.distribution(Y_test, classes, multiclass=multiclass))
        if model is None:
            if model_name == 'LSTM':
                model = self.create_LSTM_model(X_train, X_val, Y_train, Y_val, classes, callbacks, n_LSTM_layers,
                                               n_dense_layers, batch_size,
                                               activation, timesteps, n_neurons, optimizer, batch_normalization, epochs,
                                               dropout)
            else:
                model = self.create_CNN_model(X_train, X_val, Y_train, Y_val, aami_classes=classes,
                                              n_dense_layers=n_dense_layers,
                                              batch_size=batch_size, activation=activation, n_neurons=n_neurons,
                                              batch_normalization=batch_normalization, callbacks=callbacks, width=width,
                                              optimizer=optimizer,
                                              epochs=epochs)
        if validation:
            predictions = model.predict(X_val, batch_size=32)
            return eval.evaluate(predictions, Y_val, plot=False), model
        else:
            predictions = model.predict(X_test, batch_size=32)
            eval.evaluate(predictions, Y_test, classes=classes,  plot=True, title='Confusion Matrix for LSTM network')
        #self.export_model(tf.train.Saver(), [ model_name.lower() + '_1_input' ], 'dense_2/Softmax', model_name)

    def export_model( self, saver, input_node_names, output_node_name, model_name ):
        tf.train.write_graph(K.get_session().graph_def, 'out', model_name + '_graph.pbtxt')
        saver.save(K.get_session(), 'out/' + model_name + '.chkp')
        freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None,
                                  False, 'out/' + model_name + '.chkp', output_node_name,
                                  'save/restore_all', 'save/Const:0',
                                  'out/frozen_' + model_name + '.pb', True, '')
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open('out/frozen_' + model_name + '.pb', 'rb') as f:
            input_graph_def.ParseFromString(f.read())
        output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_node_names,
                                                                             [ output_node_name ],
                                                                             tf.float32.as_datatype_enum)
        with tf.gfile.FastGFile('out/opt_' + model_name + '.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('graph saved!')


if __name__ == '__main__':
    lstm = NN()
    augment = {'S':4, 'F':9}
    best_score = 0
    scores = list()
    best_window = None
    windows = [200, 300]
    for window in windows:
        score, model = lstm.beat_classification(window=window, augment=augment, validation=True, patience=10)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_window = window
            model.save('best.h5')
        K.clear_session()
    print('best')
    print(best_window)
    print(best_score)
    model = load_model('best.h5')
    lstm.beat_classification(window=best_window, augment=augment, model=model)
    plt.close()
    plt.plot(windows, scores)
    plt.ylabel('F1 score')
    plt.xlabel('window size(samples)')
    plt.legend()
    plt.show()




