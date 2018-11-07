from keras.layers import Conv2D, BatchNormalization, ELU, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential


def create_model():
    IMAGE_SIZE = (128, 128)
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=IMAGE_SIZE + [3], kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(2048))

    model.add(ELU())

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)
   return model

if __name__ == '__main__':
    weights_path = 'beatclassification/NN/CNN/ecgScratchEpoch2.hdf5'
    load_trained_model(weights_path)