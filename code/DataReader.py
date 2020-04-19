import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE
	x = []
	y = []
	
	data_directory = os.fsencode(data_dir)
	for file in os.listdir(data_directory):
		#Get the filename of the data batch file
		filename = os.fsdecode(file)
		if filename.startswith("data_batch"):
			#This is for the training files
			with open(os.path.join(data_dir, filename), 'rb') as fo:
				train_dict = pickle.load(fo, encoding='bytes')
				#[b'batch_label', b'labels', b'data', b'filenames'] are the keys in the loaded dictionary
				x.append(train_dict[b'data']) #train_dict[b'data'] is a numpy array of (1000, 3072)
				y.append(np.asarray(train_dict[b'labels'])) #train_dict[b'labels'] is a numpy array of (1000,)
		elif filename.startswith("test"):
			#This is for the test files
			with open(os.path.join(data_dir, filename), 'rb') as fo:
				test_dict = pickle.load(fo, encoding='bytes')
				x_test = test_dict[b'data']
				y_test = np.asarray(test_dict[b'labels'])

	#Concantenate everything into x_train
	x_train = x[0]
	y_train = y[0]
	for i in range(1, len(x)):
		x_train = np.concatenate((x_train, x[i]), axis = 0)
		y_train = np.concatenate((y_train, y[i]), axis = 0)
	x_train = x_train.astype(np.float32)
	y_train = y_train.astype(np.int32)
	x_test = x_test.astype(np.float32)
	y_test = y_test.astype(np.int32)	
	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid

