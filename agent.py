import sys
import os.path
import keras
import featurize
import numpy
import logging
import random
from read_for_training import erase_last_move_info

MODE_INFERENCE = "inference"
MODE_TRAINING = "training"

class Instance:
	def __init__(self, model, mode):
		self.state = None
		self.model = model
		self.mode = mode
		self.model_x_sym = None

	def predict(self):
		if self.state.turn <= 20:# and self.mode != MODE_INFERENCE:
			return []
		pred = featurize.Predictor(self.state)
		feature = pred.xs()[0]
		feature = numpy.asarray(feature)
		# if self.mode == MODE_INFERENCE:
		if True:
			rows = numpy.asarray([featurize.apply_symmetry(feature, None, sym)[0] for sym in featurize.symmetries])

			ys = self.model.predict(rows, batch_size=512)
			for i in range(len(ys)):
				_, ys[i] = featurize.apply_symmetry(None, ys[i], featurize.inverse_symmetry[featurize.symmetries[i]])
				_, t1 = featurize.apply_symmetry(None, ys[i], featurize.symmetries[i])
				_, t2 = featurize.apply_symmetry(None, t1, featurize.inverse_symmetry[featurize.symmetries[i]])
				assert numpy.array_equal(ys[i], t2)
		else:
			sym = random.choice(featurize.symmetries)
			rows = numpy.asarray([featurize.apply_symmetry(feature, None, sym)[0]])
			ys = self.model.predict(rows, batch_size=512)
			_, ys[0] = featurize.apply_symmetry(None, ys[0], featurize.inverse_symmetry[sym])
			assert len(rows) == 1 and len(ys) == 1
			self.model_x_sym = rows[0], sym
		ys = numpy.mean(ys, axis=0)
		move = pred.interpret(ys, self.mode)
		# logging.info("Turn {}: move {}".format(self.state.turn // 2, move))
		return move

	def update(self, event):
		if event['type'] == 'observe':
			self.state = featurize.state_transition(self.state, event)
			self.model_x_sym = None

