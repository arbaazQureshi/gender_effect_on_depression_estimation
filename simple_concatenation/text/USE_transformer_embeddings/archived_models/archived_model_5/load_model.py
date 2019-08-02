import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Masking, Lambda
import keras.backend as K
import keras

def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))
	X_gender = Input(shape = (2,))
	#Y = Masking()(X)

	Y = CuDNNLSTM(205, name = 'lstm_cell', return_sequences = True)(X)
	
	Y = Lambda(lambda x: K.sum(Y, axis = 1))(Y)
	Y = Dropout(rate = 0.3)(Y)

	Y = Concatenate(axis = -1)([Y, X_gender])

	Y = Dense(60, activation = 'relu', name = 'regressor_hidden_layer')(Y)
	Y = Dropout(rate = 0.3)(Y)
	
	Y = Dense(1, activation = None)(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model