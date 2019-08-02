import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Masking
import keras

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (6300, 20,))

	Y = CuDNNLSTM(30, name = 'lstm_cell')(X)
	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(35, activation = 'relu')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y_dep = Dense(1, activation = None, name = 'DLR')(Y)
	Y_gender = Dense(2, activation = 'softmax', name = 'GP')(Y)

	model = Model(inputs = X, outputs = [Y_dep, Y_gender])

	print("Created a new model.")

	return model