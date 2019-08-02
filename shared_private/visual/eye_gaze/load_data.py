import numpy as np
import pandas as pd

def load_training_data():
	
	train_set_ID_list = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]

	location = '/data/chercheurs/qureshi191/preprocessed_data/training_data/visual/eye_gaze/eye_gaze.npy'
	X = np.load(location)
	
	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/train_split_Depression_AVEC2017.csv')
	labels.set_index('Participant_ID', inplace = True)

	Y = labels['PHQ8_Score'][train_set_ID_list].values
	X_gender = np.load('/data/chercheurs/qureshi191/preprocessed_data/training_data/gender.npy')

	print("Loaded training data.")

	return (X, X_gender, Y)





def load_development_data():

	dev_set_ID_list = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 472, 476, 477, 482, 483, 484, 489, 490, 492]

	location = '/data/chercheurs/qureshi191/preprocessed_data/development_data/visual/eye_gaze/eye_gaze.npy'
	X = np.load(location)

	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/dev_split_Depression_AVEC2017.csv')
	labels.set_index('Participant_ID', inplace = True)
	
	Y = labels['PHQ8_Score'][dev_set_ID_list].values

	X_gender = np.load('/data/chercheurs/qureshi191/preprocessed_data/development_data/gender.npy')

	print("Loaded development data.")

	return (X, X_gender, Y)


def load_test_data():

	test_set_ID_list = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]

	location = '/data/chercheurs/qureshi191/preprocessed_data/test_data/visual/eye_gaze/eye_gaze.npy'
	X = np.load(location)

	labels = pd.read_csv('/data/chercheurs/qureshi191/raw_data/full_test_split.csv')
	labels.set_index('Participant_ID', inplace = True)
	
	Y = labels['PHQ_Score'][test_set_ID_list].values

	X_gender = np.load('/data/chercheurs/qureshi191/preprocessed_data/test_data/gender.npy')

	print("Loaded test data.")

	return (X, X_gender, Y)


if __name__ == "__main__":

	a = load_training_data()
	b = load_development_data()
	c = load_test_data()