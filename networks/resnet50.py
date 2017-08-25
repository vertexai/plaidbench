#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import os
import argparse
import json
import random
import sys
import time

# Import the apps
import keras.applications as kapp

# Import the sample dataset
from keras.backend.common import floatx
from keras.datasets import cifar10
from keras.layers import Input

# Load the dataset
print("Loading the data")
(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()

# Get rid of all the data except the training images (for now)
y_train_cats = None
x_test = None
y_test_cats = None

# Set a batch size
batch_size = 1

# truncate number of images
x_train = x_train[:batch_size]

# cifar10 data is only 32x32 pixels, so we must upsample by a factor of 7
# to produce the 224x224 images required by resnet.
print("Upscaling the data")
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)

# Load the model
print("Loading the model")
inputLayer = Input(shape=(224, 224, 3), dtype=floatx())
model = kapp.ResNet50(input_tensor=inputLayer)

#Prep the model and run an inital untimed batch
print("Compiling")
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print("Running initial batch")
y = model.predict(x=x_train, batch_size=batch_size)
output.contents = y

print("Warmup")
for i in range(10):
    stop_watch.start()

#Now start the clock and run 100 batches
print("Doing the main timing")
for i in range(1000):
    stop_watch.start()
    y = model.predict(x=x_train, batch_size=batch_size)
    stop_watch.stop()
    time.sleep(.025 * random.random())

