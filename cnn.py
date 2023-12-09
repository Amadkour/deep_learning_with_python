import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Dropout, \
    BatchNormalization
import math
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True)
# To support both python 2 and python 3

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
# reshape into images
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_train = X_train.reshape(X_train.shape[0], 784,1)
X_test = X_test.reshape(X_test.shape[0], 784, 1)
print(np.shape(X_train))

model = Sequential()

# model.add(Conv1D(11, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(784)))
model.add(Conv1D(32, 3, activation='relu', padding='valid', input_shape=(784, 1)))

model.add(Conv1D(9, kernel_size=3, activation='relu', strides=2, padding='valid'))


model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(5, kernel_size=9, activation='relu', padding='valid'))

model.add(Dropout(0.2))

model.add(Flatten())

# model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
#
nClasses = 10
#
# Use Keras' handy utils
y_train_k = tensorflow.keras.utils.to_categorical(y_train, num_classes=nClasses)
y_test_k = tensorflow.keras.utils.to_categorical(y_test, num_classes=nClasses)
y_train = np.array(y_train)
y_test = np.array(y_test)
batchSize = 1  # train  56,000/32 >>>>1750 batches
nEpochs = 2
#
history = model.fit(X_train, y_train_k, epochs=nEpochs, verbose=1, batch_size=16,
                    validation_data=(X_test, y_test_k))
y_predict = model.predict(X_test.astype(float))

print(y_predict[0])

x = y_predict[0]

print(np.argmax(x))
