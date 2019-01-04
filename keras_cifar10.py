__author__ = 'kevin'
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] Loading CIFAR10 data....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = trainY.reshape((testX.shape[0], 3072))

lb = LabelBinarizer()
testY = lb.fit_transform(testY)
trainY = lb.fit_transform(trainY)

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("[INFO] Training Network")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

print("[INFO]")