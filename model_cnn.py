import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model

import data_engine
import common

class model_cnn(object):

	def __init__(self):
		self.batch_size = 32
		self.num_classes = 10
		self.epochs = 20
		
		self.data_engine = data_engine.data_engine()
		self.data_engine.load()
	end

	def build(self):
		self.model = Sequential()

		'''
		self.model.add(Flatten(input_shape=(32, 32, 1)))
		self.model.add(Dense(self.num_classes)) # 10
		self.model.add(Activation('softmax'))
		'''

		## CNN is so strong that could reach 100% accuracy on test set
		self.model.add(Conv2D(16, (3, 3), padding='same', data_format='channels_last', input_shape=(32, 32, 1)))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(8, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(128))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(self.num_classes)) # 10
		self.model.add(Activation('softmax'))
		
		self.opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
		self.model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])

		plot_model(self.model, to_file='model.png', show_shapes=True)
	end
		
	def train(self):
		train_x, train_y = self.data_engine.train_set()
		test_x, test_y = self.data_engine.test_set()
		train_y = keras.utils.to_categorical(train_y, self.num_classes)
		test_y = keras.utils.to_categorical(test_y, self.num_classes)

		self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(test_x, test_y), shuffle=True)
	end
end

if __name__ == '__main__':
	model = model_cnn()
	model.build()
	# model.train()
end