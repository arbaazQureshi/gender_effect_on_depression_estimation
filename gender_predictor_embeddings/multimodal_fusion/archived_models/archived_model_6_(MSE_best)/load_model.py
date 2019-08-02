import numpy as np
import keras

from keras.models import Model, load_model
from keras.layers import Dense, Input, Concatenate, Dropout, Add, Lambda, BatchNormalization
from keras import regularizers
from keras import backend as K

from keras.engine.topology import Layer


def load_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	COVAREP = Input(shape = (95,))
	formant = Input(shape = (30,))
	text = Input(shape = (80,))
	action_units = Input(shape = (35,))
	eye_gaze = Input(shape = (25,))
	facial_landmarks = Input(shape = (85,))
	head_pose = Input(shape = (15,))

	#X_gender = Input(shape = (2,))


	common_dim = 40

	
	COVAREP_dim_adjusted = Dense(common_dim, activation = 'relu')(COVAREP)
	#COVAREP_dim_adjusted = BatchNormalization(center = False, scale = False)(COVAREP_dim_adjusted)

	formant_dim_adjusted = Dense(common_dim, activation = 'relu')(formant)
	#formant_dim_adjusted = BatchNormalization(center = False, scale = False)(formant_dim_adjusted)
	
	text_dim_adjusted = Dense(common_dim, activation = 'relu')(text)
	#text_dim_adjusted = BatchNormalization(center = False, scale = False)(text_dim_adjusted)
	
	action_units_dim_adjusted = Dense(common_dim, activation = 'relu')(action_units)
	#action_units_dim_adjusted = BatchNormalization(center = False, scale = False)(action_units_dim_adjusted)
	
	eye_gaze_dim_adjusted = Dense(common_dim, activation = 'relu')(eye_gaze)
	#eye_gaze_dim_adjusted = BatchNormalization(center = False, scale = False)(eye_gaze_dim_adjusted)
	
	facial_landmarks_dim_adjusted = Dense(common_dim, activation = 'relu')(facial_landmarks)
	#facial_landmarks_dim_adjusted = BatchNormalization(center = False, scale = False)(facial_landmarks_dim_adjusted)
	
	head_pose_dim_adjusted = Dense(common_dim, activation = 'relu')(head_pose)
	#head_pose_dim_adjusted = BatchNormalization(center = False, scale = False)(head_pose_dim_adjusted)

	
	P = Concatenate(axis = 1)([COVAREP_dim_adjusted, formant_dim_adjusted, text_dim_adjusted, action_units_dim_adjusted, eye_gaze_dim_adjusted, facial_landmarks_dim_adjusted, head_pose_dim_adjusted])

	P = Dense(150, activation = 'tanh')(P)

	alpha = Dense(7, activation = 'softmax')(P)

	F = Lambda(lambda x : alpha[:,0:1]*COVAREP_dim_adjusted + alpha[:,1:2]*formant_dim_adjusted + alpha[:,2:3]*text_dim_adjusted + alpha[:,3:4]*action_units_dim_adjusted + alpha[:,4:5]*facial_landmarks_dim_adjusted + alpha[:,5:6]*head_pose_dim_adjusted)(alpha)

	#Y = Concatenate(axis = -1)([F, X_gender])

	Y = Dense(40, activation = 'relu')(F)
	Y = Dropout(rate = 0.25)(Y)
	
	#Y = Dense(105, activation = 'relu')(Y)
	#Y = Dropout(rate = 0.2)(Y)

	Y_dep = Dense(1, activation = None, name = 'DLR')(Y)
	Y_gender = Dense(2, activation = 'softmax', name = 'GP')(Y)

	model = Model(inputs = [COVAREP, formant, text, action_units, eye_gaze, facial_landmarks, head_pose], outputs = [Y_dep, Y_gender])

	print("Created a new model.")

	return model



if(__name__ == "__main__"):
	m = load_model()