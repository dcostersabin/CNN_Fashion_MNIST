# importing the libraries

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# defining the dataset directory

BASE = os.getcwd()
TRAINING_DATASET = BASE + '/Dataset/train/fashion-mnist_train/fashion-mnist_train.csv'
TEST_DATASET = BASE + '/Dataset/test/fashion-mnist_test/fashion-mnist_test.csv'
MODEL_DIR = BASE + '/Models/myModel'


def reshape_labels(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


def testing_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_test = reshape_labels(y_test)
    x_test = np.expand_dims(x_test, -1)
    print(x_test.shape)
    x_test = x_test / 255.0
    classifier = load_model(MODEL_DIR)
    classifier.evaluate(x_test, y_test)


def training():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = reshape_labels(y_train)
    x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
    print(x_train.shape)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    classifier = build_model(x_train, y_train)
    classifier.save(MODEL_DIR)
    classifier.summary()


def build_model(training_data, training_labels):
    # initializing the CNN
    classifier = Sequential()
    # adding the convolution layer
    # classifier.add(BatchNormalization(input_shape=training_data.shape[1:]))
    classifier.add(
        Convolution2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(training_data.shape[1:])))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding another layer of conv 2d
    classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding third layer
    classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding the flatten layer
    classifier.add(Flatten())
    # adding the dense layer
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(rate=0.2))
    # adding the classifier layer
    classifier.add(Dense(units=10, activation='softmax'))
    # compiling the nn
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # feeding the data to neural network and image preprocessing
    classifier.fit(training_data, training_labels, epochs=10)
    return classifier


if __name__ == '__main__':
    # testing_model()
    if os.path.exists(MODEL_DIR):
        choice = input("Model Already Exists Do You Want To Train Y/N")
    if choice == 'Y':
        training()
    else:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        classifier = load_model(MODEL_DIR)
        dataframe = pd.read_csv(TEST_DATASET)
        data_label = dataframe.iloc[:, 0].values
        data_value = dataframe.iloc[:, 1:].values
        selection = int(input("Please Choose A Number Between 0 to " + str(len(data_label))))
        print("Selected Label", class_names[data_label[selection]])
        selection = 4
        selected_data = data_value[selection]
        selected_data = selected_data / 255.0
        selected_data = np.reshape(selected_data, (28, 28, 1))
        selected_data = np.expand_dims(selected_data, axis=0)
        prediction = classifier.predict(selected_data)

        for i in range(0, len(class_names)):
            value = float("{:.3f}".format(prediction[0][i]))
            print("The Predicted for  ", class_names[i],
                  'is', value)
