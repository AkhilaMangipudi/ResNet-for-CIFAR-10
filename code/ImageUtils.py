import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
	"""
	Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	image = preprocess_image(image, training)

	return image

def preprocess_image(image, training):
	"""Preprocess a single image of shape [height, width, depth].

	Args:
		image: An array of shape [32, 32, 3].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	if training:
		### YOUR CODE HERE
		# Resize the image to add four extra pixels on each side.
		# image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)
		image = np.pad(image, ((4,4), (4,4), (0,0)), 'constant')

		### END CODE HERE

		### YOUR CODE HERE
		# Randomly crop a [32, 32] section of the image.
		# image = tf.random_crop(image, [32, 32, 3])
		# HINT: randomly generate the upper left point of the image
		_height = np.random.randint(0, image.shape[0] - 32 + 1)
		_width = np.random.randint(0, image.shape[1] - 32 + 1)
		image = image[_height:_height + 32, _width:_width+32, :]
		### END CODE HERE

		### YOUR CODE HERE
		# Randomly flip the image horizontally.
		# image = tf.image.random_flip_left_right(image)
		image_flip = np.fliplr(image)
		#For achieveing the random flipping part, generate a random int between 0 and 100
		#If it is less than 50 flip, otherwise do not flip
		p = np.random.randint(100)
		if p < 50:
			image = image_flip
		else:
			image = image

		### END CODE HERE

	### YOUR CODE HERE
	# Subtract off the mean and divide by the standard deviation of the pixels.
	# image = tf.image.per_image_standardization(image)
	#Following the image standardization from tensor flow,
	N = image.shape[0] * image.shape[1] * image.shape[2]
	mean = np.mean(image)
	stddev = np.std(image)
	adjusted_stddev = max(stddev, 1.0/np.sqrt(N))
	image = (image - mean) / (adjusted_stddev)
	### END CODE HERE

	return image
