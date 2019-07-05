import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot
from matplotlib import animation

class FashionMNIST:
	def __init__(self):
		self.train_images = None
		self.train_labels = None
		self.test_images = None
		self.test_labels = None
		self.class_names = None
		self.model = None

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

	def build_model(self):
		self.model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=(28, 28)),
			tf.keras.layers.Dense(10, activation='softmax')
		])

		self.model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

	def train_model(self, epochs=10):
		self.model.fit(self.train_images, self.train_labels, epochs)

	def evaluate_model(self):
		test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
		print('\nTest accuracy:', test_acc)

	# TODO: Predict

	def plot_one(self, i: int):
		fig = pyplot.figure()
		pyplot.imshow(self.train_images[i])
		pyplot.colorbar()
		pyplot.grid(False)
		pyplot.show()

	def plot_one_random(self):
		fig = pyplot.figure()
		data = self.train_images
		im = pyplot.imshow(data[0])
		nx, ny = 28, 28

		def animate(_):
			ran_pic_idx = random.randint(-1, len(data) - 1)
			im.set_data(data[ran_pic_idx])
			return im, 10

		anim = animation.FuncAnimation(fig, animate, frames=nx * ny, interval=1000)
		pyplot.show()

	def plot_multiple(self):
		fig = pyplot.figure(figsize=(10, 10))

		for i in range(25):
			pyplot.subplot(5,5, i+1)
			pyplot.xticks([])
			pyplot.yticks([])
			pyplot.grid(False)
			pyplot.imshow(self.train_images[i])
			pyplot.xlabel(self.class_names[self.train_labels[i]])
		pyplot.show()
