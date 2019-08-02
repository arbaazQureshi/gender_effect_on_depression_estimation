import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Masking
import keras



def load_male_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (20000, 5,))
	#Y = Masking()(X)

	Y_male = CuDNNLSTM(35, name = 'male_specific_lstm_layer')(X)
	Y_shared = CuDNNLSTM(35, name = 'shared_lstm_layer')(X)

	Y = Concatenate(axis = -1)([Y_male, Y_shared])

	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(30, activation = 'relu', name = 'male_specific_dense_layer_1')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(1, activation = None, name = 'male_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new male model.")

	return model



def load_female_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (20000, 5,))
	#Y = Masking()(X)

	Y_female = CuDNNLSTM(35, name = 'female_specific_lstm_layer')(X)
	Y_shared = CuDNNLSTM(35, name = 'shared_lstm_layer')(X)

	Y = Concatenate(axis = -1)([Y_female, Y_shared])

	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(30, activation = 'relu', name = 'female_specific_dense_layer_1')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(1, activation = None, name = 'female_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new female model.")

	return model