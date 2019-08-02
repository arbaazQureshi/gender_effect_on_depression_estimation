from load_model import load_male_model, load_female_model
from load_data import load_training_data, load_development_data, load_test_data
import keras

from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import os
from os import path

import random

#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"

male_training_progress = []
male_development_progress = []
male_test_progress = []

female_training_progress = []
female_development_progress = []
female_test_progress = []

all_training_progress = []
all_development_progress = []
all_test_progress = []

male_model = load_male_model()
female_model = load_female_model()

male_model.compile(optimizer='adamax', loss='mse', metrics = ['mae'])
female_model.compile(optimizer='adamax', loss='mse', metrics = ['mae'])

X_train_male, Y_train_male, X_train_female, Y_train_female = load_training_data()
X_dev_male, Y_dev_male, X_dev_female, Y_dev_female = load_development_data()
X_test_male, Y_test_male, X_test_female, Y_test_female = load_test_data()

Y_train_all = np.hstack((Y_train_male, Y_train_female))
Y_dev_all = np.hstack((Y_dev_male, Y_dev_female))
Y_test_all = np.hstack((Y_test_male, Y_test_female))

min_MSE_dev_male = 10000
min_MAE_dev_male = 10000

min_MSE_dev_female = 10000
min_MAE_dev_female = 10000

min_MSE_dev_all = 10000
min_MAE_dev_all = 10000

min_MSE_test_male = 10000
min_MAE_test_male = 10000

min_MSE_test_female = 10000
min_MAE_test_female = 10000

min_MSE_test_all = 10000
min_MAE_test_all = 10000



current_epoch_number = 1
total_epoch_count = 1000

m_male = X_train_male.shape[0]
m_female = X_train_female.shape[0]

batch_male = m_male
batch_female = m_female

print("\n\n")

while(current_epoch_number <= total_epoch_count):

	if(path.exists('male_model_weights.h5')):
		male_model.load_weights('female_model_weights.h5', by_name = True)

	for epoch in range(40):

		print("\n\n\n")
		print("MALE   " * 10)
		print(total_epoch_count - current_epoch_number, "epochs left.\n\n")

		history = male_model.fit(X_train_male, Y_train_male, batch_size = batch_male, epochs = 1)
		
		Y_train_male_pred = np.squeeze(male_model.predict(X_train_male, batch_size = batch_male))
		Y_dev_male_pred = np.squeeze(male_model.predict(X_dev_male, batch_size = batch_male))
		Y_test_male_pred = np.squeeze(male_model.predict(X_test_male, batch_size = batch_male))

		Y_train_female_pred = np.squeeze(female_model.predict(X_train_female, batch_size = batch_female))
		Y_dev_female_pred = np.squeeze(female_model.predict(X_dev_female, batch_size = batch_female))
		Y_test_female_pred = np.squeeze(female_model.predict(X_test_female, batch_size = batch_female))

		Y_train_all_pred = np.hstack((Y_train_male_pred, Y_train_female_pred))
		Y_dev_all_pred = np.hstack((Y_dev_male_pred, Y_dev_female_pred))
		Y_test_all_pred = np.hstack((Y_test_male_pred, Y_test_female_pred))



		MSE_train_male = mean_squared_error(Y_train_male, Y_train_male_pred)
		MAE_train_male = mean_absolute_error(Y_train_male, Y_train_male_pred)

		MSE_train_female = mean_squared_error(Y_train_female, Y_train_female_pred)
		MAE_train_female = mean_absolute_error(Y_train_female, Y_train_female_pred)

		MSE_train_all = mean_squared_error(Y_train_all, Y_train_all_pred)
		MAE_train_all = mean_absolute_error(Y_train_all, Y_train_all_pred)



		MSE_dev_male = mean_squared_error(Y_dev_male, Y_dev_male_pred)
		MAE_dev_male = mean_absolute_error(Y_dev_male, Y_dev_male_pred)

		MSE_dev_female = mean_squared_error(Y_dev_female, Y_dev_female_pred)
		MAE_dev_female = mean_absolute_error(Y_dev_female, Y_dev_female_pred)

		MSE_dev_all = mean_squared_error(Y_dev_all, Y_dev_all_pred)
		MAE_dev_all = mean_absolute_error(Y_dev_all, Y_dev_all_pred)



		MSE_test_male = mean_squared_error(Y_test_male, Y_test_male_pred)
		MAE_test_male = mean_absolute_error(Y_test_male, Y_test_male_pred)

		MSE_test_female = mean_squared_error(Y_test_female, Y_test_female_pred)
		MAE_test_female = mean_absolute_error(Y_test_female, Y_test_female_pred)

		MSE_test_all = mean_squared_error(Y_test_all, Y_test_all_pred)
		MAE_test_all = mean_absolute_error(Y_test_all, Y_test_all_pred)



		print("Test:\t\t", MSE_test_all, MAE_test_all)
		print("Development:\t", MSE_dev_all, MAE_dev_all)
		print("Train:\t\t", MSE_train_all, MAE_train_all)


		male_training_progress.append([current_epoch_number, MSE_train_male, MAE_train_male])
		male_development_progress.append([current_epoch_number, MSE_dev_male, MAE_dev_male])
		male_test_progress.append([current_epoch_number, MSE_test_male, MAE_test_male])

		female_training_progress.append([current_epoch_number, MSE_train_female, MAE_train_female])
		female_development_progress.append([current_epoch_number, MSE_dev_female, MAE_dev_female])
		female_test_progress.append([current_epoch_number, MSE_test_female, MAE_test_female])

		all_training_progress.append([current_epoch_number, MSE_train_all, MAE_train_all])
		all_development_progress.append([current_epoch_number, MSE_dev_all, MAE_dev_all])
		all_test_progress.append([current_epoch_number, MSE_test_all, MAE_test_all])


		if(MSE_dev_all < min_MSE_dev_all):
			
			min_MSE_dev_all = MSE_dev_all
			male_model.save_weights('opton_MSE_dev_all_male_model.h5')
			female_model.save_weights('opton_MSE_dev_all_female_model.h5')
			print("BEST MSE DEV ALL MODEL!\n\n")

			with open('development_MSE_all_best.txt', 'w') as f:
				
				f.write('Min development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_all < min_MAE_dev_all):
			
			min_MAE_dev_all = MAE_dev_all
			male_model.save_weights('opton_MAE_dev_all_male_model.h5')
			female_model.save_weights('opton_MAE_dev_all_female_model.h5')
			print("BEST MAE DEV ALL MODEL!\n\n")

			with open('development_MAE_all_best.txt', 'w') as f:
				
				f.write('Min development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_all < min_MSE_test_all):
			
			min_MSE_test_all = MSE_test_all
			male_model.save_weights('opton_MSE_test_all_male_model.h5')
			female_model.save_weights('opton_MSE_test_all_female_model.h5')
			print("BEST MSE TEST ALL MODEL!\n\n")

			with open('test_MSE_all_best.txt', 'w') as f:
				
				f.write('Min test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_all < min_MAE_test_all):
			
			min_MAE_test_all = MAE_test_all
			male_model.save_weights('opton_MAE_test_all_male_model.h5')
			female_model.save_weights('opton_MAE_test_all_female_model.h5')
			print("BEST MAE TEST ALL MODEL!\n\n")

			with open('test_MAE_all_best.txt', 'w') as f:
				
				f.write('Min test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	




		if(MSE_dev_male < min_MSE_dev_male):
			
			min_MSE_dev_male = MSE_dev_male
			male_model.save_weights('opton_MSE_dev_male_male_model.h5')
			female_model.save_weights('opton_MSE_dev_male_female_model.h5')
			print("BEST MSE DEV MALE MODEL!\n\n")

			with open('development_MSE_male_best.txt', 'w') as f:

				f.write('Min development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n\n')


				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_male < min_MAE_dev_male):
			
			min_MAE_dev_male = MAE_dev_male
			male_model.save_weights('opton_MAE_dev_male_male_model.h5')
			female_model.save_weights('opton_MAE_dev_male_female_model.h5')
			print("BEST MAE DEV MALE MODEL!\n\n")

			with open('development_MAE_male_best.txt', 'w') as f:

				f.write('Min development MAE male:\t' + str(MAE_dev_male) + '\n')
				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')


				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_male < min_MSE_test_male):
			
			min_MSE_test_male = MSE_test_male
			male_model.save_weights('opton_MSE_test_male_male_model.h5')
			female_model.save_weights('opton_MSE_test_male_female_model.h5')
			print("BEST MSE TEST MALE MODEL!\n\n")

			with open('test_MSE_male_best.txt', 'w') as f:
				
				f.write('Min test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_male < min_MAE_test_male):
			
			min_MAE_test_male = MAE_test_male
			male_model.save_weights('opton_MAE_test_male_male_model.h5')
			female_model.save_weights('opton_MAE_test_male_female_model.h5')
			print("BEST MAE TEST MALE MODEL!\n\n")

			with open('test_MAE_male_best.txt', 'w') as f:
				
				f.write('Min test MAE male:\t\t' + str(MAE_test_male) + '\n')
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	





		if(MSE_dev_female < min_MSE_dev_female):
			
			min_MSE_dev_female = MSE_dev_female
			male_model.save_weights('opton_MSE_dev_female_male_model.h5')
			female_model.save_weights('opton_MSE_dev_female_female_model.h5')
			print("BEST MSE DEV FEMALE MODEL!\n\n")

			with open('development_MSE_female_best.txt', 'w') as f:

				f.write('Min development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')



				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n\n')


				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')

				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_female < min_MAE_dev_female):
			
			min_MAE_dev_female = MAE_dev_female
			male_model.save_weights('opton_MAE_dev_female_male_model.h5')
			female_model.save_weights('opton_MAE_dev_female_female_model.h5')
			print("BEST MAE DEV FEMALE MODEL!\n\n")

			with open('development_MAE_female_best.txt', 'w') as f:

				f.write('Min development MAE female:\t' + str(MAE_dev_female) + '\n')
				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')



				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')

				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_female < min_MSE_test_female):
			
			min_MSE_test_female = MSE_test_female
			male_model.save_weights('opton_MSE_test_female_male_model.h5')
			female_model.save_weights('opton_MSE_test_female_female_model.h5')
			print("BEST MSE TEST FEMALE MODEL!\n\n")

			with open('test_MSE_female_best.txt', 'w') as f:
				
				f.write('Min test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n\n')



				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_female < min_MAE_test_female):
			
			min_MAE_test_female = MAE_test_female
			male_model.save_weights('opton_MAE_test_female_male_model.h5')
			female_model.save_weights('opton_MAE_test_female_female_model.h5')
			print("BEST MAE TEST FEMALE MODEL!\n\n")

			with open('test_MAE_female_best.txt', 'w') as f:
				
				f.write('Min test MAE female:\t\t' + str(MAE_test_female) + '\n')
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n\n')



				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	

		current_epoch_number = current_epoch_number + 1	






	np.savetxt('male_training_progress.csv', np.array(male_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('male_development_progress.csv', np.array(male_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('male_test_progress.csv', np.array(male_test_progress), fmt='%.4f', delimiter=',')

	np.savetxt('female_training_progress.csv', np.array(female_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('female_development_progress.csv', np.array(female_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('female_test_progress.csv', np.array(female_test_progress), fmt='%.4f', delimiter=',')

	np.savetxt('all_training_progress.csv', np.array(all_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('all_development_progress.csv', np.array(all_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('all_test_progress.csv', np.array(all_test_progress), fmt='%.4f', delimiter=',')






	if(path.exists('female_model_weights.h5')):
		female_model.load_weights('male_model_weights.h5', by_name = True)

	for epoch in range(40):

		print("\n\n\n")
		print("FEMALE   " * 10)
		print(total_epoch_count - current_epoch_number, "epochs left.\n\n")

		history = female_model.fit(X_train_female, Y_train_female, batch_size = batch_male, epochs = 1)
		
		Y_train_male_pred = np.squeeze(male_model.predict(X_train_male, batch_size = batch_male))
		Y_dev_male_pred = np.squeeze(male_model.predict(X_dev_male, batch_size = batch_male))
		Y_test_male_pred = np.squeeze(male_model.predict(X_test_male, batch_size = batch_male))

		Y_train_female_pred = np.squeeze(female_model.predict(X_train_female, batch_size = batch_female))
		Y_dev_female_pred = np.squeeze(female_model.predict(X_dev_female, batch_size = batch_female))
		Y_test_female_pred = np.squeeze(female_model.predict(X_test_female, batch_size = batch_female))

		Y_train_all_pred = np.hstack((Y_train_male_pred, Y_train_female_pred))
		Y_dev_all_pred = np.hstack((Y_dev_male_pred, Y_dev_female_pred))
		Y_test_all_pred = np.hstack((Y_test_male_pred, Y_test_female_pred))



		MSE_train_male = mean_squared_error(Y_train_male, Y_train_male_pred)
		MAE_train_male = mean_absolute_error(Y_train_male, Y_train_male_pred)

		MSE_train_female = mean_squared_error(Y_train_female, Y_train_female_pred)
		MAE_train_female = mean_absolute_error(Y_train_female, Y_train_female_pred)

		MSE_train_all = mean_squared_error(Y_train_all, Y_train_all_pred)
		MAE_train_all = mean_absolute_error(Y_train_all, Y_train_all_pred)



		MSE_dev_male = mean_squared_error(Y_dev_male, Y_dev_male_pred)
		MAE_dev_male = mean_absolute_error(Y_dev_male, Y_dev_male_pred)

		MSE_dev_female = mean_squared_error(Y_dev_female, Y_dev_female_pred)
		MAE_dev_female = mean_absolute_error(Y_dev_female, Y_dev_female_pred)

		MSE_dev_all = mean_squared_error(Y_dev_all, Y_dev_all_pred)
		MAE_dev_all = mean_absolute_error(Y_dev_all, Y_dev_all_pred)



		MSE_test_male = mean_squared_error(Y_test_male, Y_test_male_pred)
		MAE_test_male = mean_absolute_error(Y_test_male, Y_test_male_pred)

		MSE_test_female = mean_squared_error(Y_test_female, Y_test_female_pred)
		MAE_test_female = mean_absolute_error(Y_test_female, Y_test_female_pred)

		MSE_test_all = mean_squared_error(Y_test_all, Y_test_all_pred)
		MAE_test_all = mean_absolute_error(Y_test_all, Y_test_all_pred)


		print("Test:\t\t", MSE_test_all, MAE_test_all)
		print("Development:\t", MSE_dev_all, MAE_dev_all)
		print("Train:\t\t", MSE_train_all, MAE_train_all)


		male_training_progress.append([current_epoch_number, MSE_train_male, MAE_train_male])
		male_development_progress.append([current_epoch_number, MSE_dev_male, MAE_dev_male])
		male_test_progress.append([current_epoch_number, MSE_test_male, MAE_test_male])

		female_training_progress.append([current_epoch_number, MSE_train_female, MAE_train_female])
		female_development_progress.append([current_epoch_number, MSE_dev_female, MAE_dev_female])
		female_test_progress.append([current_epoch_number, MSE_test_female, MAE_test_female])

		all_training_progress.append([current_epoch_number, MSE_train_all, MAE_train_all])
		all_development_progress.append([current_epoch_number, MSE_dev_all, MAE_dev_all])
		all_test_progress.append([current_epoch_number, MSE_test_all, MAE_test_all])



		if(MSE_dev_all < min_MSE_dev_all):
			
			min_MSE_dev_all = MSE_dev_all
			male_model.save_weights('opton_MSE_dev_all_male_model.h5')
			female_model.save_weights('opton_MSE_dev_all_female_model.h5')
			print("BEST MSE DEV ALL MODEL!\n\n")

			with open('development_MSE_all_best.txt', 'w') as f:
				
				f.write('Min development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_all < min_MAE_dev_all):
			
			min_MAE_dev_all = MAE_dev_all
			male_model.save_weights('opton_MAE_dev_all_male_model.h5')
			female_model.save_weights('opton_MAE_dev_all_female_model.h5')
			print("BEST MAE DEV ALL MODEL!\n\n")

			with open('development_MAE_all_best.txt', 'w') as f:
				
				f.write('Min development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_all < min_MSE_test_all):
			
			min_MSE_test_all = MSE_test_all
			male_model.save_weights('opton_MSE_test_all_male_model.h5')
			female_model.save_weights('opton_MSE_test_all_female_model.h5')
			print("BEST MSE TEST ALL MODEL!\n\n")

			with open('test_MSE_all_best.txt', 'w') as f:
				
				f.write('Min test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_all < min_MAE_test_all):
			
			min_MAE_test_all = MAE_test_all
			male_model.save_weights('opton_MAE_test_all_male_model.h5')
			female_model.save_weights('opton_MAE_test_all_female_model.h5')
			print("BEST MAE TEST ALL MODEL!\n\n")

			with open('test_MAE_all_best.txt', 'w') as f:
				
				f.write('Min test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	




		if(MSE_dev_male < min_MSE_dev_male):
			
			min_MSE_dev_male = MSE_dev_male
			male_model.save_weights('opton_MSE_dev_male_male_model.h5')
			female_model.save_weights('opton_MSE_dev_male_female_model.h5')
			print("BEST MSE DEV MALE MODEL!\n\n")

			with open('development_MSE_male_best.txt', 'w') as f:

				f.write('Min development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n\n')


				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_male < min_MAE_dev_male):
			
			min_MAE_dev_male = MAE_dev_male
			male_model.save_weights('opton_MAE_dev_male_male_model.h5')
			female_model.save_weights('opton_MAE_dev_male_female_model.h5')
			print("BEST MAE DEV MALE MODEL!\n\n")

			with open('development_MAE_male_best.txt', 'w') as f:

				f.write('Min development MAE male:\t' + str(MAE_dev_male) + '\n')
				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')


				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_male < min_MSE_test_male):
			
			min_MSE_test_male = MSE_test_male
			male_model.save_weights('opton_MSE_test_male_male_model.h5')
			female_model.save_weights('opton_MSE_test_male_female_model.h5')
			print("BEST MSE TEST MALE MODEL!\n\n")

			with open('test_MSE_male_best.txt', 'w') as f:
				
				f.write('Min test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_male < min_MAE_test_male):
			
			min_MAE_test_male = MAE_test_male
			male_model.save_weights('opton_MAE_test_male_male_model.h5')
			female_model.save_weights('opton_MAE_test_male_female_model.h5')
			print("BEST MAE TEST MALE MODEL!\n\n")

			with open('test_MAE_male_best.txt', 'w') as f:
				
				f.write('Min test MAE male:\t\t' + str(MAE_test_male) + '\n')
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')



				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')
				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	





		if(MSE_dev_female < min_MSE_dev_female):
			
			min_MSE_dev_female = MSE_dev_female
			male_model.save_weights('opton_MSE_dev_female_male_model.h5')
			female_model.save_weights('opton_MSE_dev_female_female_model.h5')
			print("BEST MSE DEV FEMALE MODEL!\n\n")

			with open('development_MSE_female_best.txt', 'w') as f:

				f.write('Min development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')



				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n\n')


				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')

				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')

		

		if(MAE_dev_female < min_MAE_dev_female):
			
			min_MAE_dev_female = MAE_dev_female
			male_model.save_weights('opton_MAE_dev_female_male_model.h5')
			female_model.save_weights('opton_MAE_dev_female_female_model.h5')
			print("BEST MAE DEV FEMALE MODEL!\n\n")

			with open('development_MAE_female_best.txt', 'w') as f:

				f.write('Min development MAE female:\t' + str(MAE_dev_female) + '\n')
				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')
				
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n\n')



				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n')
				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')
				
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n\n')



				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')

				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')
		


		if(MSE_test_female < min_MSE_test_female):
			
			min_MSE_test_female = MSE_test_female
			male_model.save_weights('opton_MSE_test_female_male_model.h5')
			female_model.save_weights('opton_MSE_test_female_female_model.h5')
			print("BEST MSE TEST FEMALE MODEL!\n\n")

			with open('test_MSE_female_best.txt', 'w') as f:
				
				f.write('Min test MSE female:\t\t' + str(MSE_test_female) + '\n')
				f.write('Corresponding test MAE female:\t\t' + str(MAE_test_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n\n')



				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')
				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')
				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')



		if(MAE_test_female < min_MAE_test_female):
			
			min_MAE_test_female = MAE_test_female
			male_model.save_weights('opton_MAE_test_female_male_model.h5')
			female_model.save_weights('opton_MAE_test_female_female_model.h5')
			print("BEST MAE TEST FEMALE MODEL!\n\n")

			with open('test_MAE_female_best.txt', 'w') as f:
				
				f.write('Min test MAE female:\t\t' + str(MAE_test_female) + '\n')
				f.write('Corresponding test MSE female:\t\t' + str(MSE_test_female) + '\n')

				f.write('Corresponding training MSE female:\t\t' + str(MSE_train_female) + '\n')
				f.write('Corresponding training MAE female:\t\t' + str(MAE_train_female) + '\n')

				f.write('Corresponding development MSE female:\t\t' + str(MSE_dev_female) + '\n')
				f.write('Corresponding development MAE female:\t' + str(MAE_dev_female) + '\n\n')



				f.write('Corresponding test MAE male:\t\t' + str(MAE_test_male) + '\n')
				f.write('Corresponding test MSE male:\t\t' + str(MSE_test_male) + '\n')

				f.write('Corresponding training MSE male:\t\t' + str(MSE_train_male) + '\n')
				f.write('Corresponding training MAE male:\t\t' + str(MAE_train_male) + '\n')

				f.write('Corresponding development MSE male:\t\t' + str(MSE_dev_male) + '\n')
				f.write('Corresponding development MAE male:\t' + str(MAE_dev_male) + '\n\n')



				f.write('Corresponding test MAE all:\t\t' + str(MAE_test_all) + '\n')
				f.write('Corresponding test MSE all:\t\t' + str(MSE_test_all) + '\n')

				f.write('Corresponding development MSE all:\t\t' + str(MSE_dev_all) + '\n')
				f.write('Corresponding development MAE all:\t' + str(MAE_dev_all) + '\n')
				
				f.write('Corresponding training MSE all:\t\t' + str(MSE_train_all) + '\n')
				f.write('Corresponding training MAE:\t\t' + str(MAE_train_all) + '\n\n')


				
				f.write('Epoch number:\t\t' + str(current_epoch_number) + '\n')	

		current_epoch_number = current_epoch_number + 1



	np.savetxt('male_training_progress.csv', np.array(male_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('male_development_progress.csv', np.array(male_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('male_test_progress.csv', np.array(male_test_progress), fmt='%.4f', delimiter=',')

	np.savetxt('female_training_progress.csv', np.array(female_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('female_development_progress.csv', np.array(female_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('female_test_progress.csv', np.array(female_test_progress), fmt='%.4f', delimiter=',')

	np.savetxt('all_training_progress.csv', np.array(all_training_progress), fmt='%.4f', delimiter=',')
	np.savetxt('all_development_progress.csv', np.array(all_development_progress), fmt='%.4f', delimiter=',')
	np.savetxt('all_test_progress.csv', np.array(all_test_progress), fmt='%.4f', delimiter=',')

	print("\n\n")