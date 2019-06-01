import h5py
import keras
from keras.models import Model
from keras.layers import Input, Lambda, RepeatVector, Reshape, Activation, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from read_for_training import parse_csv_row_xy, parse_csv_row_xvalue
import keras.backend as K

import featurize
import sys
import random
import os.path
import numpy as np
import math


flen = featurize.FEATURE_LENGTH
n = featurize.MAXSIZE
pertile = featurize.FEATURES_PER_TILE
hlen = featurize.FEATURE_HEADER_LENGTH
classes = 4
ylen = n * n * classes

class SplitLeft:
	def __init__(self, posn):
		self.__name__ = "SplitLeft"
		self.posn = posn
	def __call__(self, x):
		return x[:, :self.posn]
class SplitRight:
	def __init__(self, posn):
		self.__name__ = "SplitRight"
		self.posn = posn
	def __call__(self, x):
		return x[:, self.posn:]

def blank_policy_network(opt='adam'):
	inp = Input(shape=(flen,))
	head = Lambda(SplitLeft(hlen), output_shape=(hlen,))(inp)
	head = RepeatVector(n*n)(head)
	head = Reshape((n, n, hlen))(head)
	tail = Lambda(SplitRight(hlen), output_shape=(flen-hlen,))(inp)
	tail = Reshape((n, n, pertile))(tail)
	img = Concatenate(3)([head, tail])
	filters = 256
	while img.shape[1] != 1:
		img = Conv2D(filters, 3, padding='same', activation='relu')(img)
		# img = BatchNormalization(axis=3)(img)
		img = AveragePooling2D(padding='same')(img)
	img = Flatten()(img)
	img = RepeatVector(n*n)(img)
	img = Reshape((n, n, filters))(img)
	img = Concatenate(3)([img, tail])
	for _ in range(14):
		img = Conv2D(filters, 3, padding='same', activation='relu')(img)
		# img = BatchNormalization(axis=3)(img)
	img = Conv2D(classes, 1, padding='same', activation=None)(img)
	img = Reshape((ylen,))(img)
	img = Activation('softmax')(img)

	m = Model(inp, img)
	m.compile(
		optimizer=opt,
		loss='categorical_crossentropy',
	    metrics=['accuracy'])
	return m

value_hlen = featurize.VALUE_FEATURE_HEADER_LENGTH
value_flen = featurize.VALUE_FEATURE_LENGTH
value_pertile = featurize.VALUE_FEATURES_PER_TILE

def blank_value_network(opt='adam'):
	inp = Input(shape=(value_flen,))
	base_head = Lambda(SplitLeft(value_hlen), output_shape=(value_hlen,))(inp)
	head = base_head
	head = RepeatVector(n*n)(head)
	head = Reshape((n, n, value_hlen))(head)
	tail = Lambda(SplitRight(value_hlen), output_shape=(value_flen-value_hlen,))(inp)
	tail = Reshape((n, n, value_pertile))(tail)
	img = Concatenate(3)([head, tail])
	filters = 4 
	for _ in range(2):
		img = Conv2D(filters, 1, padding='same', activation='relu')(img)
		img = BatchNormalization(axis=3)(img)
	img = Reshape((n * n, filters))(img)
	img = Lambda(lambda x: K.mean(x, axis=1))(img)
	img = Concatenate(1)([img, base_head])
	img = Dense(32, activation='relu')(img)
	img = Dense(32, activation='relu')(img)
	img = Dense(1, activation='tanh')(img)

	m = Model(inp, img)
	m.compile(
		optimizer=opt,
		loss='mean_squared_error')
	return m


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(sys.argv, file=sys.stderr)
		print("Please provide a path to store the trained neural net.", file=sys.stderr)
		sys.exit(2)
	inc = 1
	filename = sys.argv[1]
	firstsave = "%s-%d.h5" % (filename, inc)
	if os.path.exists(firstsave):
		print("The file `%s` exists. Please delete it manually and rerun the script." % firstsave, file=sys.stderr)
		sys.exit(2)

	random.seed(12345)

	print("Reading list of features from stdin...", file=sys.stderr)
	features = [x.strip() for x in sys.stdin]
	print("...read %d features." % len(features), file=sys.stderr)

	isValue = 'value' in features[0]
	print("Training %s network." % ('VALUE' if isValue else 'POLICY'), file=sys.stderr)

	if isValue:
		m = blank_value_network()
	else:
		m = blank_policy_network()
	if len(sys.argv) >= 3:
		m.load_weights(sys.argv[2])
		print("Loaded existing model from `%s`." % sys.argv[2], file=sys.stderr)

	training_samples = int(len(features) * 0.9)
	validation_samples = len(features) - training_samples

	if isValue:
		training_data = features[validation_samples:]
		validation_data = features[:validation_samples]
	else:
		random.shuffle(features)
		training_data = features[:training_samples]
		validation_data = features[training_samples:]
	

	batch_size = 32

	crnt_flen = value_flen if isValue else flen
	crnt_ylen = 1 if isValue else ylen

	if isValue:
		mu = [0] * value_hlen
		sigma = [1] * value_hlen

	def take_sample(body):
		path = random.choice(body)
		with open(path) as f:
			for line in f:
				if isValue:
					x, y = parse_csv_row_xvalue(line)
					sym = random.choice(featurize.symmetries)
					# x, _ = featurize.apply_symmetry(x, None, sym)
					for i in range(len(mu)):
						x[i] = (x[i] - mu[i]) / sigma[i]
				else:
					x, y = parse_csv_row_xy(line, True) # WE ARE ZEROING OUT THE LAST MOVE
					sym = random.choice(featurize.symmetries)
					x, y = featurize.apply_symmetry(x, y, sym)
				return x, y
		assert False

	if isValue:
		print("Normalizing features...", file=sys.stderr)
		ss = 2048
		samples = [take_sample(training_data)[0] for i in range(ss)]
		for i in range(value_hlen):
			for s in samples:
				mu[i] += s[i]
			mu[i] /= len(samples)
			for s in samples:
				sigma[i] += (s[i] - mu[i]) ** 2
			sigma[i] /= len(samples)
			sigma[i] = max(0.0001, math.sqrt(sigma[i]))
		print("...mu={}, sigma={}.".format(mu, sigma), file=sys.stderr)

	def heuristic(x):
		from math import tanh
		return tanh((x[2] - x[3]) / 50)

	def take_batch(body):
		xs = np.empty((batch_size, crnt_flen))
		ys = np.empty((batch_size, crnt_ylen))
		for i in range(batch_size):
			x, y = take_sample(body)
			assert len(x) == crnt_flen
			assert len(y) == crnt_ylen
			xs[i, :] = x
			ys[i, :] = y
		terms = []
		for i in range(batch_size):
			h = heuristic(xs[i])
			terms.append((h - ys[i]) ** 2)
	#	print(sum(terms) / len(terms))
		return xs, ys

	def training_gen():
		while True:
			yield take_batch(training_data)
	def validation_gen():
		while True:
			yield take_batch(validation_data)

	print("Initiating training with %d (x8) training samples and %d (x8) validation samples." %
		(training_samples, validation_samples), file=sys.stderr)
	crnt = 0
	while True:
		crnt += inc
		print("Epoch %d." % crnt, file=sys.stderr)
	#	px, py = take_batch(validation_data)
	#	my = m.predict(px)
	#	print(my)
	#	print(py - my)
	#	a = py - my
	#	for i in range(len(a)):
	#		a[i] **= 2
	#	print(np.mean(a))
		per_epoch = max(1, training_samples // batch_size)
		if isValue:
			per_epoch = max(1, per_epoch // 8)
		m.fit_generator(
			generator=training_gen(),
			steps_per_epoch=per_epoch,
			epochs=1,
			validation_data=validation_gen(),
			validation_steps=(max(1, validation_samples // batch_size)))
		name = "%s-%d.h5" % (filename, crnt)
		m.save("%s-%d.h5" % (filename, crnt))
