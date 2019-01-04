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
testX = testX.astype("float")/255.0
