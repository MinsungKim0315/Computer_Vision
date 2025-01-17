import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
# change 28X28 map to 1-dimension
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# change uint8 to float32 for calculation | the /255.0 is for changing [0,255] to [0,1]
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
# express the correct values(y) to one-hot code
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Construct multi-layer perceptron
mlp = Sequential()
# add: build up layer, Dense: edges are fully connected
# 784 nodes in input layer, 512 nodes in hidden layer, activation function: hyperbolic tangent
mlp.add(Dense(units=512, activation='tanh', input_shape=(784,)))
# 10 nodes in output layer, activation function: softmax
mlp.add(Dense(units=10, activation='softmax'))

# Learn the model
# loss: choose loss function, optimizer: choose optimizer, metrics: choose metric for getting performance
mlp.compile(loss='MSE', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
# batch_size: set the size of mini batch, epochs: set the number of epochs, 
# validation_data: gives command to use x_test, y_test to get performance, 
# verbose=2: for every 1 epoch, =0: don't show, =1: show progress bar
mlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)

res = mlp.evaluate(x_test, y_test, verbose=0)   # evaluate: save the performances
print('Accuracy=', res[1]*100)  # res[1] is 'accuracy'
