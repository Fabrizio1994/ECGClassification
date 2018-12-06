from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, Conv2D, Flatten
from keras.models import Sequential
from keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scikitplot as splt
from time import time
from beatclassification.data_visualization import data_visualization
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score
from beatclassification.Preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

dv = data_visualization()
prep = Preprocessing()


class Evaluation(Callback):
    def __init__( self, plot=False, patience=None, plot_every=1 ):
        super().__init__()
        self.plot = plot
        self.epoch_count = 0
        self.plot_every = plot_every
        self.patience = patience
        self.patience_count = 0
        self.prev = 0

    def on_train_begin( self, logs={} ):
        self.val_s_recall = [ ]
        self.val_s_precision = [ ]
        self.val_f_recall = [ ]
        self.val_f_precision = [ ]

    def on_epoch_end( self, epoch, logs={} ):
        self.epoch_count += 1
        val_predict = (np.asarray(self.model.predict(self.validation_data[ 0 ]))).round()
        val_targ = self.validation_data[ 1 ]
        _val_class_recall = recall_score(val_targ, val_predict, average=None)
        _val_class_precision = precision_score(val_targ, val_predict, average=None)
        class_s_recall = _val_class_recall[ 1 ]
        self.val_s_recall.append(class_s_recall)
        class_s_precision = _val_class_precision[ 1 ]
        self.val_s_precision.append(class_s_precision)
        class_f_recall = _val_class_recall[ 3 ]
        self.val_f_recall.append(class_f_recall)
        class_f_precision = _val_class_precision[ 3 ]
        self.val_f_precision.append(class_f_precision)
        average = (class_s_precision + class_s_recall + class_f_recall + class_f_precision) / 4

        if average < self.prev:
            self.patience_count += 1
            if self.patience_count == self.patience:
                self.model.stop_training = True
        else:
            self.prev = average
            self.patience_count = 0

        if self.plot:
            plt.close()
            plt.plot(self.val_s_recall, label='class S recall')
            plt.plot(self.val_s_precision, label='class S precision')
            plt.plot(self.val_f_precision, label='class F precision')
            plt.plot(self.val_f_recall, label='class F recall')
            plt.plot()
            plt.legend()
            if self.epoch_count % self.plot_every == 0:
                plt.show()
                self.evaluate(val_predict, val_targ)
            else:
                self.evaluate(val_predict, val_targ, plot=False)
        return

    def evaluate( self, predictions, Y_test, title=None, one_hot=True, plot=True ):
        if one_hot:
            predictions = list(map(lambda x: np.argmax(x), predictions))
            Y_test = list(map(lambda x: np.argmax(x), Y_test))
        if plot:
            splt.metrics.plot_confusion_matrix(Y_test, predictions, normalize=True, title=title,
                                               title_fontsize='small')
        print("per class precision")
        precision = precision_score(Y_test, predictions, average=None)
        print(precision)
        print("average precision")
        av_precision = np.mean(precision)
        print(av_precision)
        print('per class recall')
        recall = recall_score(Y_test, predictions, average=None)
        print(recall)
        print('average recall')
        av_recall = np.mean(recall)
        print(av_recall)
        if plot:
            plt.show()
            plt.close()
        return precision[ 1 ], recall[ 1 ], precision[ 3 ], recall[ 3 ]


class NN:

    def preprocess( self, classes, augment, reduce, timesteps, train_size, split='vertical', channels=None,
                    standardize=True, aami=True, model=None, filtered=False ):
        if channels is None:
            channels = [ 0 ]
        if split == 'vertical':
            X_train, Y_train, X_val, Y_val, X_test, Y_test = prep.preprocess_split(train_size=train_size,
                                                                                   classes=classes,
                                                                                   timesteps=timesteps,
                                                                                   channels=channels,
                                                                                   standardize=standardize,
                                                                                   aami=aami, model=model,
                                                                                   filtered=filtered)
        # de Chazal horizontal split
        else:
            train_dataset = [ '106', '112', '122', '201', '223', '230', "108", "109", "115", "116", "118", "119", "124",
                              "205", "207", "208", "209", "215", '101', '114', '203', '220' ]
            test_dataset = [ "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213",
                             "214", "219",
                             "221", "222", "228", "231", "232", "233", "234" ]
            X_train, Y_train = prep.preprocess(train_dataset, channels, model=model, timesteps=timesteps)
            X_test, Y_test = prep.preprocess(test_dataset, channels, model=model, timesteps=timesteps)
            _, X_val, _, Y_val = train_test_split(X_test, Y_test, test_size=0.1)
        dv.distribution(Y_train, classes=classes)
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
        print('train distribution')
        print(dv.distribution(Y_train, classes))
        print('val distribution')
        print(dv.distribution(Y_val, classes))
        print('test distribution')
        print(dv.distribution(Y_test, classes))
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def create_LSTM_model( self, X_train, X_val, Y_train, Y_val, aami_classes, callbacks, n_LSTM_layers, n_dense_layers,
                           batch_size,
                           activation, timesteps, n_neurons, optimizer, batch_normalization, epochs, dropout ):
        model = Sequential()
        input_shape = (timesteps, X_train.shape[ 2 ])
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

    def beat_classification( self, classes=None, aami=True ):
        if classes is None:
            classes = [ 'N', 'S', 'V', 'F' ]
        augment = {'S': 3, 'F': 7}
        reduce = None
        train_size = 0.5
        channels = [0]
        n_dense_layers = 1
        batch_size = 32
        standardize = True
        activation = 'relu'
        timesteps = 5
        n_neurons = 256
        filtered = True
        width = 2
        model_name = 'LSTM'
        epochs = 100
        metrics = Evaluation()
        dropout = 0.5
        n_LSTM_layers = 1
        callbacks = [ EarlyStopping(patience=10, restore_best_weights=True) ]
        optimizer = SGD()
        batch_normalization = True
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.preprocess(classes, augment, reduce,
                                                                         timesteps, train_size,
                                                                         channels=channels,
                                                                         standardize=standardize, aami=aami,
                                                                         model=model_name, split='vertical',
                                                                         filtered=filtered)
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
        predictions = model.predict(X_test, batch_size=32)
        metrics.evaluate(predictions, Y_test, title='result on test set', plot=True)
        self.export_model(tf.train.Saver(), [ model_name.lower() + '_1_input' ], 'dense_2/Softmax', model_name)
        model.save('lstm.h5')

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
    lstm.beat_classification()
