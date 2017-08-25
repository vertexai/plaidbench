#!/usr/bin/env python

# Copyright 2017 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import time
import random

# Import the apps
import keras.applications as kapp

# cifar10 data is 1/7th the size vgg19 needs in the spatial dimensions,
# but if we upscale we can use it
from keras.datasets import cifar10

from keras.layers import Input
from keras.backend.common import floatx

# Load the dataset
print("Loading the data")
(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()

# Get rid of all the data except the training images (for now
y_train_cats = None
x_test = None
y_test_cats = None

# Set a batch size
batch_size = 1

# truncate number of images
x_train = x_train[:batch_size]

# Upscale image size by a factor of 7
print("Upscaling the data")
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)

# Load the model
print("Loading the model")
inputLayer = Input(shape=(224, 224, 3), dtype=floatx())
model = kapp.VGG19(input_tensor=inputLayer)

# Prep the model and run an initial un-timed batch
print("Compiling")
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch")
y = model.predict(x=x_train, batch_size=batch_size)
output.contents = y

print("Warmup")
for i in range(10):
    stop_watch.start()

# Now start the clock and run 100 batches
print("Doing the main timing")
for i in range(1000):
    stop_watch.start()
    y = model.predict(x=x_train, batch_size=batch_size)
    stop_watch.stop()
    time.sleep(.025 * random.random())

