from keras.models import Model
from load_model import load_model
from load_data import load_training_data, load_development_data, load_test_data
import keras

import numpy as np
import os
from os import path

import random

#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"

X_train, X_train_gender, Y_train = load_training_data()
X_dev, X_dev_gender, Y_dev = load_development_data()
X_test, X_test_gender, Y_test = load_test_data()

model = load_model()
model.load_weights('best.h5')

model = Model(inputs = model.inputs, outputs = model.layers[-4].output)

X_train_features = model.predict(X_train)
X_dev_features = model.predict(X_dev)
X_test_features = model.predict(X_test)

np.save('text_gen_training_set_features.npy', X_train_features)
np.save('text_gen_development_set_features.npy', X_dev_features)
np.save('text_gen_test_set_features.npy', X_test_features)

print(X_train_features.shape, X_dev_features.shape, X_test_features.shape)