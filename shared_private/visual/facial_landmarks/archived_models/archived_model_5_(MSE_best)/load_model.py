import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Masking
import keras

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (10000, 204,))
	X_gender = Input(shape = (2,))
	#Y = Masking()(X)

	Y = CuDNNLSTM(130, name = 'lstm_cell')(X)
	Y = Dropout(rate = 0.3)(Y)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(85, activation = 'relu')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model