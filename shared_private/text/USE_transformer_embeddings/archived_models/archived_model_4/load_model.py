import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Lambda
import keras
import keras.backend as K



def load_male_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))
	#Y = Masking()(X)

	Y_male = CuDNNLSTM(120, name = 'male_specific_lstm_layer', return_sequences = True)(X)
	Y_shared = CuDNNLSTM(120, name = 'shared_lstm_layer', return_sequences = True)(X)

	Y_male = Lambda(lambda x: K.sum(Y_male, axis = 1))(Y_male)
	Y_shared = Lambda(lambda x: K.sum(Y_shared, axis = 1))(Y_shared)

	Y = Concatenate(axis = -1)([Y_male, Y_shared])

	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(80, activation = 'relu', name = 'male_specific_dense_layer_1')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None, name = 'male_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new male model.")

	return model



def load_female_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))
	#Y = Masking()(X)

	Y_female = CuDNNLSTM(120, name = 'female_specific_lstm_layer', return_sequences = True)(X)
	Y_shared = CuDNNLSTM(120, name = 'shared_lstm_layer', return_sequences = True)(X)

	Y_female = Lambda(lambda x: K.sum(Y_female, axis = 1))(Y_female)
	Y_shared = Lambda(lambda x: K.sum(Y_shared, axis = 1))(Y_shared)

	Y = Concatenate(axis = -1)([Y_female, Y_shared])

	Y = Dropout(rate = 0.3)(Y)

	Y = Dense(80, activation = 'relu', name = 'female_specific_dense_layer_1')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None, name = 'female_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new female model.")

	return model