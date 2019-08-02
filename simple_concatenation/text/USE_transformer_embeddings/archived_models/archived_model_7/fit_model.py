from load_model import load_model
from load_data import load_training_data, load_development_data, load_test_data
import keras

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="3,5,6,7"

training_progress = []
development_progress = []
test_progress = []

model = load_model()
model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

X_train, X_train_gender, Y_train = load_training_data()
X_dev, X_dev_gender, Y_dev = load_development_data()
X_test, X_test_gender, Y_test = load_test_data()

min_mse_dev = 10000
min_mae_dev = 10000

min_mse_test = 10000
min_mae_test = 10000

current_epoch_number = 1
total_epoch_count = 300

m = X_train.shape[0]
batch_size_list = list(range(1, m))

print("\n\n")

while(current_epoch_number <= total_epoch_count):
	
	print((str(total_epoch_count - current_epoch_number)+' ')*20)

	#batch_size = random.choice(batch_size_list)
	#batch_size = int(m/2)
	batch_size = m
	print("Batch size is", batch_size)
	
	history = model.fit([X_train, X_train_gender], Y_train, batch_size = batch_size, epochs = 1)
	
	loss_dev = model.evaluate([X_dev, X_dev_gender], Y_dev, batch_size = batch_size)
	loss_test = model.evaluate([X_test, X_test_gender], Y_test, batch_size = batch_size)

	loss_train = [history.history['loss'][0], history.history['mean_absolute_error'][0]]

	print("Test:\t\t", loss_test[0], loss_test[1])
	print("Development:\t", loss_dev[0], loss_dev[1])
	print("Train:\t\t", loss_train[0], loss_train[1])

	if(loss_dev[0] < min_mse_dev):
		
		min_mse_dev = loss_dev[0]
		model.save_weights('opton_dev_MSE.h5')
		print("BEST DEV MSE MODEL!\n\n")

		with open('development_MSE_best.txt', 'w') as f:
			f.write('Min development MSE:\t\t' + str(loss_dev[0]) + '\n')
			f.write('Corresponding development MAE:\t' + str(loss_dev[1]) + '\n')
			f.write('Corresponding test MSE:\t\t' + str(loss_test[0]) + '\n')
			f.write('Corresponding test MAE:\t\t' + str(loss_test[1]) + '\n')
			f.write('Corresponding training MSE:\t\t' + str(loss_train[0]) + '\n')
			f.write('Corresponding training MAE:\t\t' + str(loss_train[1]) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(loss_dev[1] < min_mae_dev):
		
		min_mae_dev = loss_dev[1]
		model.save_weights('opton_dev_MAE.h5')
		print("BEST DEV MAE MODEL!\n\n")

		with open('development_MAE_best.txt', 'w') as f:
			f.write('Min development MAE:\t\t' + str(loss_dev[1]) + '\n')
			f.write('Corresponding development MSE:\t' + str(loss_dev[0]) + '\n')
			f.write('Corresponding test MSE:\t\t' + str(loss_test[0]) + '\n')
			f.write('Corresponding test MAE:\t\t' + str(loss_test[1]) + '\n')
			f.write('Corresponding training MSE:\t\t' + str(loss_train[0]) + '\n')
			f.write('Corresponding training MAE:\t\t' + str(loss_train[1]) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(loss_test[0] < min_mse_test):
		
		min_mse_test = loss_test[0]
		model.save_weights('opton_test_MSE.h5')
		print("BEST TEST MSE MODEL!\n\n")

		with open('test_MSE_best.txt', 'w') as f:
			f.write('Min test MSE:\t\t' + str(loss_test[0]) + '\n')
			f.write('Corresponding test MAE:\t\t' + str(loss_test[1]) + '\n')
			f.write('Corresponding development MSE:\t' + str(loss_dev[0]) + '\n')
			f.write('Corresponding development MAE:\t\t' + str(loss_dev[1]) + '\n')
			f.write('Corresponding training MSE:\t\t' + str(loss_train[0]) + '\n')
			f.write('Corresponding training MAE:\t\t' + str(loss_train[1]) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	if(loss_test[1] < min_mae_test):
		
		min_mae_test = loss_test[1]
		model.save_weights('opton_test_MAE.h5')
		print("BEST TEST MAE MODEL!\n\n")

		with open('test_MAE_best.txt', 'w') as f:
			f.write('Min test MAE:\t\t' + str(loss_test[1]) + '\n')
			f.write('Corresponding test MSE:\t\t' + str(loss_test[0]) + '\n')
			f.write('Corresponding development MSE:\t' + str(loss_dev[0]) + '\n')
			f.write('Corresponding development MAE:\t\t' + str(loss_dev[1]) + '\n')
			f.write('Corresponding training MSE:\t\t' + str(loss_train[0]) + '\n')
			f.write('Corresponding training MAE:\t\t' + str(loss_train[1]) + '\n')
			f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

	training_progress.append([current_epoch_number, loss_train[0], loss_train[1]])
	development_progress.append([current_epoch_number, loss_dev[0], loss_dev[1]])
	test_progress.append([current_epoch_number, loss_test[0], loss_test[1]])

	np.savetxt('training_progress.csv', np.array(training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('development_progress.csv', np.array(development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('test_progress.csv', np.array(test_progress), fmt='%.4f', delimiter=',')

	current_epoch_number = current_epoch_number + 1
	print("\n\n")