from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
import numpy as np
import random


def linear_unit(x, W, b):
	return tf.matmul(x, W) + b


class ModelPart0:
	def __init__(self):
		"""
        This model class contains a single layer network similar to Assignment 1.
        """

		self.batch_size = 64
		self.num_classes = 2
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		input = 32 * 32 * 3
		output = 2
		self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W1")
		self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B1")

		self.trainable_variables = [self.W1, self.B1]

	def call(self, inputs):
		"""
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		# this reshape "flattens" the image data
		inputs = np.reshape(inputs, [inputs.shape[0], -1])
		x = linear_unit(inputs, self.W1, self.B1)
		return x


class ModelPart1:
	def __init__(self):
		"""
        This model class contains a single layer network similar to Assignment 1.
        """

		self.batch_size = 64
		self.num_classes = 2
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		input = 32 * 32 * 3
		output = 2
		middle_output = 256
		self.W1 = tf.Variable(tf.random.truncated_normal([input, middle_output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W1")
		self.B1 = tf.Variable(tf.random.truncated_normal([1, middle_output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B1")

		self.W2 = tf.Variable(tf.random.truncated_normal([middle_output, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W2")
		self.B2 = tf.Variable(tf.random.truncated_normal([1, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B2")

		self.trainable_variables = [self.W1, self.B1, self.W2, self.B2]

	def call(self, inputs):
		"""
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		inputs = tf.reshape(inputs, [inputs.shape[0], -1])

		# First layer: Linear + ReLU activation
		x = tf.nn.relu(tf.matmul(inputs, self.W1) + self.B1)

		# Output layer: Linear transformation
		logits = tf.matmul(x, self.W2) + self.B2
		return logits


class ModelPart3:
	def __init__(self):
		"""
        This model class contains a single layer network similar to Assignment 1.
        """
		self.batch_size = 64
		self.num_classes = 2
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

		input = 65536
		output = self.num_classes
		middle_output = 32
		middle_output2 = 256
		self.l1filters = tf.Variable(tf.random.truncated_normal([3, 3, 3, 64],
		                                                 dtype=tf.float32,
		                                                 stddev=1e-1),
		                      name="l1filters")
		self.l2filters = tf.Variable(tf.random.truncated_normal([3, 3, 64, 64],
		                                                        dtype=tf.float32,
		                                                        stddev=1e-1),
		                             name="l1filters")

		self.W1 = tf.Variable(tf.random.truncated_normal([input, middle_output2],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W1")
		self.B1 = tf.Variable(tf.random.truncated_normal([1, middle_output2],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B1")

		self.W2 = tf.Variable(tf.random.truncated_normal([middle_output2, middle_output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W2")
		self.B2 = tf.Variable(tf.random.truncated_normal([1, middle_output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B2")

		self.W3 = tf.Variable(tf.random.truncated_normal([middle_output, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="W3")
		self.B3 = tf.Variable(tf.random.truncated_normal([1, output],
		                                                 dtype=tf.float32,
		                                                 stddev=0.1),
		                      name="B3")

		self.trainable_variables = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.l1filters, self.l2filters]

	def call(self, inputs):
		"""
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)


		#first layer -> convolution and relu
		l1 = tf.nn.relu(tf.nn.conv2d(inputs, self.l1filters, [1, 1, 1, 1], padding='SAME'))
		l2 = tf.nn.relu(tf.nn.conv2d(l1, self.l2filters, [1, 1, 1, 1], padding='SAME'))
		pooled_inputs = tf.nn.max_pool2d (l2,ksize=(2, 2), strides=(1, 1),
		                                  padding="SAME")
		inputs = tf.reshape(l2, [tf.shape(pooled_inputs)[0], -1])

		# second layer: Linear + ReLU activation
		l3 = tf.nn.relu(tf.matmul(inputs, self.W1) + self.B1)

		# Output layer: Linear transformation
		l4 = tf.matmul(l3, self.W2) + self.B2

		logits = linear_unit(l4, self.W3, self.B3)
		return logits

def loss(logits, labels):
	"""
    Calculates the cross-entropy loss after one forward pass.
    :param logits: during training, a matrix of shape (batch_size, self.num_classes)
    containing the result of multiple convolution and feed forward layers
    Softmax is applied in this function.
    :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
    :return: the loss of the model as a Tensor
    """
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def accuracy(logits, labels):
	"""
    Calculates the model's prediction accuracy by comparing
    logits to correct labels – no need to modify this.
    :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

    NOTE: DO NOT EDIT

    :return: the accuracy of the model as a Tensor
    """
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
	"""
    Trains the model on all the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    """
	indices = tf.range(start=0, limit=len(train_inputs), dtype=tf.int32)
	indices = tf.random.shuffle(indices)
	inputs = tf.gather(train_inputs, indices)
	labels = tf.gather(train_labels, indices)

	for start in range(0, len(inputs), model.batch_size):
		end = start + model.batch_size
		batch_inputs = inputs[start:end]
		batch_labels = labels[start:end]

		with tf.GradientTape() as tape:
			logits = model.call(batch_inputs)
			loss_value = loss(logits, batch_labels)

		gradients = tape.gradient(loss_value, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
	"""
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """
	return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (10, num_classes)
    :param image_labels: the labels from get_data(), shape (10, num_classes)
    :param first_label: the name of the first class, "dog"
    :param second_label: the name of the second class, "cat"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
		ax.imshow(image_inputs[ind], cmap="Greys")
		pl = first_label if predicted_labels[ind] == 0.0 else second_label
		al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
		ax.set(title="PL: {}\nAL: {}".format(pl, al))
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		ax.tick_params(axis='both', which='both', length=0)
	plt.show()


CLASS_CAT = 3
CLASS_DOG = 5


def main(cifar10_data_folder):
	"""
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    25 epochs.
    You should receive a final accuracy on the testing examples for cat and dog
    of ~60% for Part1 and ~70% for Part3.
    :return: None
    """
	EPOCHS = 25
	FIRST_CLASS = 3
	SECOND_CLASS = 5
	#model = ModelPart0()
	#model = ModelPart1()
	model = ModelPart3()
	train_inputs, train_labels = get_data(cifar10_data_folder + "train", FIRST_CLASS, SECOND_CLASS)
	test_inputs, test_labels = get_data(cifar10_data_folder + "test", FIRST_CLASS, SECOND_CLASS)
	for epoch in range(EPOCHS):
		train(model, train_inputs, train_labels)
		accur = test(model, train_inputs, train_labels)
		logits = model.call(train_inputs)
		epoch_loss = loss(logits, train_labels)
		print(f"Epoch: {epoch + 1} accuracy={accur * 100:.0f}% loss: {epoch_loss}")
	print(f"Test result accuracy: {test(model, test_inputs, test_labels) * 100:.0f}%")
	#visualize_results(test_inputs[0:10], model.call(test_inputs[0:10]), test_labels[0:10], FIRST_CLASS, SECOND_CLASS)
	return


if __name__ == '__main__':
	cifar_data_folder = 'CIFAR_data/'
	with tf.device('/GPU:0'):
		main(cifar_data_folder)
