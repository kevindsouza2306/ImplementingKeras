__author__ = 'kevin'
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] Loading MNIST (full) dataset....")
datasets = datasets.fetch_mldata("MNIST Original")
data = datasets.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, datasets.target, test_size=0.25)\

lb = LabelBinarizer
testY = lb.fit_transform(testY)
trainY = lb.fit_transform(trainY)
