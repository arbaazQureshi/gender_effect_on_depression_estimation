from load_model import load_model
from load_data import load_training_data, load_development_data, load_test_data
import keras

import numpy as np
import os
from os import path

import random

#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"

training_progress = []
development_progress = []
test_progress = []

loss_funcs = {'DLR' : 'mse', 'GP' : 'categorical_crossentropy'}
loss_weights = {'DLR' : 1.0, 'GP' : 1.0}
metrics = {'DLR' : 'mae', 'GP' : 'accuracy'}

model = load_model()
model.compile(optimizer='adamax', loss=loss_funcs,  loss_weights = loss_weights, metrics = metrics)


COVAREP_train, formant_train, text_train, action_units_train, eye_gaze_train, facial_landmarks_train, head_pose_train, Y_train, Y_train_gender = load_training_data()
COVAREP_dev, formant_dev, text_dev, action_units_dev, eye_gaze_dev, facial_landmarks_dev, head_pose_dev, Y_dev, Y_dev_gender = load_development_data()
COVAREP_test, formant_test, text_test, action_units_test, eye_gaze_test, facial_landmarks_test, head_pose_test, Y_test, Y_test_gender = load_test_data()

min_DLR_MSE_dev = 10000
min_DLR_MAE_dev = 10000
min_GP_CE_dev = 10000
max_GP_accuracy_dev = -1

min_DLR_MSE_test = 10000
min_DLR_MAE_test = 10000
min_GP_CE_test = 10000
max_GP_accuracy_test = -1

current_epoch_number = 1
total_epoch_count = 300

m = COVAREP_train.shape[0]
batch_size_list = list(range(1, m))

print("\n\n")

while(current_epoch_number <= total_epoch_count):
	
	print((str(total_epoch_count - current_epoch_number)+' ')*20)

	#batch_size = random.choice(batch_size_list)
	#batch_size = int(m/4)
	batch_size = m
	print("Batch size is", batch_size)
	
	history = model.fit([COVAREP_train, formant_train, text_train, action_units_train, eye_gaze_train, facial_landmarks_train, head_pose_train], [Y_train, Y_train_gender], batch_size = batch_size, epochs = 1)
	loss_train, DLR_MSE_train, GP_CE_train, DLR_MAE_train, GP_accuracy_train = [history.history['loss'][0], history.history['DLR_loss'][0], history.history['GP_loss'][0], history.history['DLR_mean_absolute_error'][0], history.history['GP_acc'][0]]
	
	loss_dev, DLR_MSE_dev, GP_CE_dev, DLR_MAE_dev, GP_accuracy_dev = model.evaluate([COVAREP_dev, formant_dev, text_dev, action_units_dev, eye_gaze_dev, facial_landmarks_dev, head_pose_dev], [Y_dev, Y_dev_gender], batch_size = batch_size)
	loss_test, DLR_MSE_test, GP_CE_test, DLR_MAE_test, GP_accuracy_test = model.evaluate([COVAREP_test, formant_test, text_test, action_units_test, eye_gaze_test, facial_landmarks_test, head_pose_test], [Y_test, Y_test_gender], batch_size = batch_size)

	print("Test:\t\t", DLR_MSE_test, DLR_MAE_test, GP_CE_test, GP_accuracy_test, loss_test)
	print("Development:\t", DLR_MSE_dev, DLR_MAE_dev, GP_CE_dev, GP_accuracy_dev, loss_dev)
	print("Train:\t\t", DLR_MSE_train, DLR_MAE_train, GP_CE_train, GP_accuracy_train, loss_train)

	if(DLR_MSE_dev < min_DLR_MSE_dev):
		
		min_DLR_MSE_dev = DLR_MSE_dev
		model.save_weights('opton_dev_DLR_MSE.h5')
		print("BEST DEV_DLR_MSE MODEL!\n\n")

		with open('development_DLR_MSE_best.txt', 'w') as f:
			
			f.write('Min development DLR_MSE:\t\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding development DLR_MAE:\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding development GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding development GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding development total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test DLR_MSE:\t\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')
			
			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if(DLR_MAE_dev < min_DLR_MAE_dev):
		
		min_DLR_MAE_dev = DLR_MAE_dev
		model.save_weights('opton_dev_DLR_MAE.h5')
		print("BEST DEV_DLR_MAE MODEL!\n\n")

		with open('development_DLR_MAE_best.txt', 'w') as f:
			
			f.write('Min development DLR_MAE:\t\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding development DLR_MSE:\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding development GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding development GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding development total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test DLR_MSE:\t\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if(GP_CE_dev < min_GP_CE_dev):
		
		min_GP_CE_dev = GP_CE_dev
		model.save_weights('opton_dev_GP_CE.h5')
		print("BEST DEV_GP_CE MODEL!\n\n")

		with open('development_GP_CE_best.txt', 'w') as f:
			
			f.write('Min development GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding development GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding development DLR_MAE:\t\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding development DLR_MSE:\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding development total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test DLR_MSE:\t\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if((GP_accuracy_dev > max_GP_accuracy_dev) or (GP_accuracy_dev == max_GP_accuracy_dev and GP_CE_dev < min_GP_CE_dev)):
		
		max_GP_accuracy_dev = GP_accuracy_dev
		model.save_weights('opton_dev_GP_accuracy.h5')
		print("BEST DEV_GP_accuracy MODEL!\n\n")

		with open('development_GP_accuracy_best.txt', 'w') as f:
			
			f.write('Max development GP_accuracy:\t\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding development GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding development DLR_MAE:\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding development DLR_MSE:\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding development total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding test DLR_MSE:\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test GP_CE:\t\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')




	if(DLR_MSE_test < min_DLR_MSE_test):
		
		min_DLR_MSE_test = DLR_MSE_test
		model.save_weights('opton_test_DLR_MSE.h5')
		print("BEST TEST_DLR_MSE MODEL!\n\n")

		with open('test_DLR_MSE_best.txt', 'w') as f:
			
			f.write('Min test DLR_MSE:\t\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding dev DLR_MSE:\t\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding dev DLR_MAE:\t\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding dev GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding dev GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding dev total_loss:\t' + str(loss_dev) + '\n\n')
			
			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if(DLR_MAE_test < min_DLR_MAE_test):
		
		min_DLR_MAE_test = DLR_MAE_test
		model.save_weights('opton_test_DLR_MAE.h5')
		print("BEST TEST_DLR_MAE MODEL!\n\n")

		with open('test_DLR_MAE_best.txt', 'w') as f:
			
			f.write('Min test DLR_MAE:\t\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test DLR_MSE:\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding dev DLR_MSE:\t\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding dev DLR_MAE:\t\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding dev GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding dev GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding dev total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if(GP_CE_test < min_GP_CE_test):
		
		min_GP_CE_test = GP_CE_test
		model.save_weights('opton_test_GP_CE.h5')
		print("BEST TEST_GP_CE MODEL!\n\n")

		with open('test_GP_CE_best.txt', 'w') as f:
			
			f.write('Min test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test GP_accuracy:\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test DLR_MSE:\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding dev DLR_MSE:\t\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding dev DLR_MAE:\t\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding dev GP_CE:\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding dev GP_accuracy:\t' + str(GP_accuracy_dev) + '\n')
			f.write('Corresponding dev total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



	if((GP_accuracy_test > max_GP_accuracy_test) or (GP_accuracy_test == max_GP_accuracy_test and GP_CE_test < min_GP_CE_test)):
		
		max_GP_accuracy_test = GP_accuracy_test
		model.save_weights('opton_test_GP_accuracy.h5')
		print("BEST TEST_GP_accuracy MODEL!\n\n")

		with open('test_GP_accuracy_best.txt', 'w') as f:
			
			f.write('Max test GP_accuracy:\t\t' + str(GP_accuracy_test) + '\n')
			f.write('Corresponding test GP_CE:\t' + str(GP_CE_test) + '\n')
			f.write('Corresponding test DLR_MAE:\t' + str(DLR_MAE_test) + '\n')
			f.write('Corresponding test DLR_MSE:\t' + str(DLR_MSE_test) + '\n')
			f.write('Corresponding test total_loss:\t' + str(loss_test) + '\n\n')

			f.write('Corresponding dev DLR_MSE:\t' + str(DLR_MSE_dev) + '\n')
			f.write('Corresponding dev DLR_MAE:\t' + str(DLR_MAE_dev) + '\n')
			f.write('Corresponding dev GP_CE:\t\t' + str(GP_CE_dev) + '\n')
			f.write('Corresponding dev GP_accuracy:\t' + str(GP_accuracy_dev) + '\n\n')
			f.write('Corresponding dev total_loss:\t' + str(loss_dev) + '\n\n')

			f.write('Corresponding training DLR_MSE:\t\t' + str(DLR_MSE_train) + '\n')
			f.write('Corresponding training DLR_MAE:\t\t' + str(DLR_MAE_train) + '\n')
			f.write('Corresponding training GP_CE:\t' + str(GP_CE_train) + '\n')
			f.write('Corresponding training GP_accuracy:\t' + str(GP_accuracy_train) + '\n')
			f.write('Corresponding training total_loss:\t' + str(loss_train) + '\n\n')

			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	training_progress.append([DLR_MSE_train, DLR_MAE_train, GP_CE_train, GP_accuracy_train, loss_train])
	development_progress.append([DLR_MSE_dev, DLR_MAE_dev, GP_CE_dev, GP_accuracy_dev, loss_dev])
	test_progress.append([DLR_MSE_test, DLR_MAE_test, GP_CE_test, GP_accuracy_test, loss_test])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('development_progress.csv', np.array(development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('test_progress.csv', np.array(test_progress), fmt='%.4f', delimiter=',')

	current_epoch_number = current_epoch_number + 1
	print("\n\n")