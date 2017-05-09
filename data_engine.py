import os
import os.path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import common

class data_engine(object):

	def __init__(self):
		base_dir = './Nosie/'
		self.train_dir = base_dir + 'TRAIN/'
		self.test_dir = base_dir + 'TEST/'
	end

	def load(self):
		# load train data
		train_x, train_y = [], []
		i, disp_freq = 0, 1e10

		for parent, dirnames, filenames in os.walk(self.train_dir):
			for filename in filenames:
				img = mpimg.imread(os.path.join(parent, filename)).astype('float32')

				# cut 3-channel img to 1-channel
				if len(img.shape) == 3:
					img = img[:, :, 0]
				end

				# cut img -> (32, 32)
				img = img[:32, :32]

				# assure img shape
				try:
					assert img.shape == (32, 32)
				except:
					print img.shape
				end

				# regularization
				if np.max(img) > 1:
					img /= 255.
				
				if parent.endswith('digits'):
					label = int(filename[0])
				elif parent.endswith('hjk_picture'):
					try:
						label = int(filename[filename.find('.') + 1])
					except:
						label = int(filename[0])
					end
				elif parent.endswith('Wanjin'):
					try:
						label = int(filename[filename.find('-') + 1])
					except:
						label = int(filename[0])
					end
				else: # number
					label = int(filename[filename.find('.') + 1])
				end
					
				train_x.append(img)
				train_y.append(label)
				i += 1
				if i % disp_freq == 0:
					print os.path.join(parent, filename)
					plt.imshow(img, cmap='gray')
					plt.title(label)
					plt.show()
				end
			end
		end

		self.train_x = np.asarray(train_x)
		self.train_y = np.asarray(train_y)
		print 'loaded train set, shape', self.train_x.shape
		# print np.max(self.train_x) # 1.0
		print 'loaded train label, shape', self.train_y.shape

		# load test data

		test_x, test_y = [], []
		i, disp_freq = 0, 1e20
		for parent, dirnames, filenames in os.walk(self.test_dir):
			for filename in filenames:

				# There are 5 .png files that matplotlib cannot read.
				try:
					img = mpimg.imread(os.path.join(parent, filename)).astype('float32')
				except:
					# print 'imread error', filename
					continue
				end

				# cut 3-channel img to 1-channel
				if len(img.shape) == 3:
					img = img[:, :, 0]
				end

				# cut img -> (32, 32)
				img = img[:32, :32]

				# assure img shape
				try:
					assert img.shape == (32, 32)
				except:
					print img.shape
				end

				# regularization
				if np.max(img) > 1:
					img /= 255.

				label = int(filename[filename.find('[') + 1])

				test_x.append(img)
				test_y.append(label)
				i += 1
				if i % disp_freq == 0:
					print os.path.join(parent, filename)
					plt.imshow(img, cmap='gray')
					plt.title(label)
					plt.show()
				end
			end
		end

		self.test_x = np.asarray(test_x)
		self.test_y = np.asarray(test_y)
		print 'loaded test set, shape', self.test_x.shape
		# print np.max(self.test_x) # 1.0
		print 'loaded test label, shape', self.test_y.shape

	end # def

end # class 

if __name__ == '__main__':
	engine = data_engine()
	engine.load()
end
