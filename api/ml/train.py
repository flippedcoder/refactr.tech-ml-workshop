import json
import os
import pickle
import ssl
import yaml

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# set SSL for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

# load in the dataset
data = tf.keras.datasets.cifar100.load_data(label_mode="fine")

# load in the hyperparams
params = yaml.safe_load(open("params.yaml"))

# check data structure
# print(data[0])

# split the data into train and test datasets
(x_train, y_train), (x_test, y_test) = data

# visualize data
# plt.figure(figsize=(5,5))
# plt.imshow(x_train[32])
# plt.show()

# number of labels that are there for the data
num_classes = params["num_classes"]

# build the CNN (convolutional neural network) model
model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(32,32,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

# add the optimizer
learning_rate = params["learning_rate"]
opt = Adam(learning_rate=learning_rate)

# compile model
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# define hyperparams
epochs = params["num_epochs"]
shuffle = params["shuffle"]

# train the model
cnn_history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), shuffle=shuffle)

# evaluate the model
acc = cnn_history.history["accuracy"]
val_acc = cnn_history.history["val_accuracy"]
loss = cnn_history.history["loss"]
val_loss = cnn_history.history["val_loss"]

epochs_range = range(epochs)

# evaluation plots
plt.figure(figsize=(15,15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.show()

# save CNN model
model_output = "api/ml/models/model.pkl"

with (open(model_output, "wb")) as fd:
    pickle.dump(model, fd)