import numpy as np
import matplotlib.pyplot as plt

import data_engine


'''
	train_x: [n, 32, 32]
	train_y: [n, ]
'''
def noise_cancellation(train_x, train_y, test_x, test_y):

	prior_probs = np.zeros((10, 32, 32))

	# calculate prior probabilities
	for i in range(len(train_y)):
		prior_probs[train_y[i]] += train_x[i]

	for k in range(10):
		prior = prior_probs[k]
		prior /= prior.sum()
		prior[prior < 0.001] = 0
		prior /= prior.sum()
		prior_probs[k] = prior
	
	for k in range(10):

		continue

		fig = plt.imshow(prior_probs[k], cmap='gray')
		# plt.colorbar()
		
		# print prior_probs[k]
		plt.imsave('%d.png'%k, prior_probs[k])
		# plt.show()

	for i in range(len(train_y)):
		ori_max = np.amax(train_x[i])
		img = np.multiply(train_x[i], prior_probs[train_y[i]])
		img = img / np.amax(img) * ori_max
		train_x[i] = img

		continue

		f, axarr = plt.subplots(1, 3)
		axarr[0].imshow(train_x[i], cmap='gray')
		axarr[1].imshow(prior_probs[train_y[i]], cmap='gray')
		axarr[2].imshow(img, cmap='gray')
		plt.title('%d'%train_y[i])
		plt.show()
	
	for i in range(len(test_y)):

		maxv, maxk = -1e20, -1
		for k in range(10):
			likelihood = np.multiply(test_x[i], prior_probs[k]).sum()
			if likelihood > maxv:
				maxv = likelihood
				maxk = k

		ori_max = np.amax(test_x[i])
		img = np.multiply(test_x[i], prior_probs[maxk])
		img = img / np.amax(img) * ori_max
		test_x[i] = img		
	
	return train_x, train_y, test_x, test_y


if __name__ == '__main__':

	engine = data_engine.data_engine()
	engine.load()

	train_x, train_y = engine.train_x.reshape((-1, 32, 32)), engine.train_y

	noise_cancellation(train_x, train_y, None, None)
