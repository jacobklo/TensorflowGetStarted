import tensorflow as tf
import numpy as np
from matplotlib import pyplot


class FashionMNIST:
	def __init__(self):
		self.train_images = None
		self.train_labels = None
		self.test_images = None
		self.test_labels = None
		self.class_names = None

		self.load()
		self.prepare()

	def load(self):
		fashion_mnist = tf.keras.datasets.fashion_mnist
		(self.train_images, self.train_labels), (self.test_images, self.test_labels) = \
			fashion_mnist.load_data()

		self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
							'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	def prepare(self):
		self.train_images = self.train_images / 255
		self.test_images = self.test_images / 255

	def plot_one(self, i: int):
		fig = pyplot.figure()
		fig.imshow(self.train_images[i])
		fig.colorbar()
		fig.grid(False)
		fig.show()
		# TODO : dynmic update specific plot

