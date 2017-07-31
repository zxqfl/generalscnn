import tensorflow as tf
import numpy as np
import featurize
import sys
import os
import random
from read_for_training import parse_csv_row_xy
from stats import StatisticReporter

side_length = featurize.MAXSIZE
policy_input_length = featurize.FEATURE_LENGTH
policy_input_channels = featurize.FEATURES_PER_TILE
policy_header_length = featurize.FEATURE_HEADER_LENGTH
policy_output_channels = 4
policy_output_length = side_length * side_length * policy_output_channels

class TFModel:
	def __init__(self):
		# def normalize(x, ident=""):
		# 	depth = x.shape[-1]
		# 	mean = tf.get_variable("mean" + ident, [depth], initializer=)
		def init_policy():
			inp = tf.placeholder("float32", [None, policy_input_length])
			header = inp[:, :policy_header_length]
			map_img = inp[:, policy_header_length:]

			header_img = tf.reshape(header, [-1, 1, 1, policy_header_length])
			header_img = tf.tile(header_img, [1, side_length, side_length, 1])
			map_img = tf.reshape(map_img, [-1, side_length, side_length, policy_input_channels])
			base_img = tf.concat([header_img, map_img], 3)
			assert base_img.shape[1] == side_length
			assert base_img.shape[2] == side_length
			assert base_img.shape[3] == featurize.FEATURE_HEADER_LENGTH + featurize.FEATURES_PER_TILE

			base_img = tf.contrib.layers.batch_norm(inputs=base_img, updates_collections=None)

			main_filters = 256
			aux_filters = 32
			layer_count = 15

			with tf.variable_scope("convmaxpool"):
				img = base_img
				index = 0
				while img.shape[1] != 1:
					in_channels = img.shape[3]
					out_channels = aux_filters
					kernel_size = 3
					weights = tf.get_variable("kernel-%d" % index, [kernel_size, kernel_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
					img = tf.nn.conv2d(
						input=img,
						filter=weights,
						strides=[1, 1, 1, 1],
						padding="SAME")
					img = tf.nn.relu(img)
					if index >= 3:
						img = tf.nn.max_pool(
							value=img,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding="SAME")
					index += 1
				maxpool_result = img
				maxpool_img = tf.tile(maxpool_result, [1, side_length, side_length, 1])

			with tf.variable_scope("convavgpool"):
				img = base_img
				index = 0
				while img.shape[1] != 1:
					in_channels = img.shape[3]
					out_channels = aux_filters
					kernel_size = 3
					weights = tf.get_variable("kernel-%d" % index, [kernel_size, kernel_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
					img = tf.nn.conv2d(
						input=img,
						filter=weights,
						strides=[1, 1, 1, 1],
						padding="SAME")
					img = tf.nn.relu(img)
					if index >= 3:
						img = tf.nn.avg_pool(
							value=img,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding="SAME")
					index += 1
				avgpool_result = img
				avgpool_img = tf.tile(avgpool_result, [1, side_length, side_length, 1])

			base_img = tf.concat([base_img, avgpool_img, maxpool_img], 3)

			with tf.variable_scope("convlocal"):
				img = base_img
				for i in range(layer_count):
					if i+1 == layer_count:
						kernel_size = 1
					elif i == 0:
						kernel_size = 5
					else:
						kernel_size = 3
					# img = tf.contrib.layers.convolution2d(
					# 	inputs=img,
					# 	num_outputs=main_filters if i+1 < layer_count else policy_output_channels,
					# 	kernel_size=kernel_size,
					# 	activation_fn=tf.nn.relu if i+1 < layer_count else None,
					in_channels = img.shape[3]
					out_channels = main_filters if i+1 < layer_count else policy_output_channels
					weights = tf.get_variable("kernel-%d" % i, [kernel_size, kernel_size, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
					img = tf.nn.conv2d(
						input=img,
						filter=weights,
						strides=[1, 1, 1, 1],
						padding="SAME")
					if i+1 != layer_count:
						img = tf.nn.relu(img)

			logits = tf.reshape(img, [-1, policy_output_length])
			policy = tf.nn.softmax(logits)
			y_true = tf.placeholder("int32", [None, policy_output_length])
			sl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				logits=logits,
				labels=y_true))
			correct_prediction = tf.equal(tf.argmax(policy, 1), tf.argmax(y_true, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

			self.sl_policy_loss = sl_loss
			self.sl_policy_y_true = y_true
			self.policy_input = inp
			self.policy = policy
			self.sl_policy_accuracy = accuracy
			self.debug_logits = logits

		init_policy()

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(sys.argv, file=sys.stderr)
		print("Please provide a path to store the trained neural net.", file=sys.stderr)
		sys.exit(2)
	inc = 1
	filename = sys.argv[1]
	firstsave = "%s-%d" % (filename, inc)
	if os.path.exists(firstsave):
		print("The file `%s` exists. Please delete it manually and rerun the script." % firstsave, file=sys.stderr)
		sys.exit(2)

	random.seed(12345)

	print("Reading list of features from stdin...", file=sys.stderr)
	features = [x.strip() for x in sys.stdin]
	print("...read %d features." % len(features), file=sys.stderr)

	assert 'value' not in features[0]

	training_samples = int(len(features) * 0.9)
	validation_samples = len(features) - training_samples

	random.shuffle(features)
	training_data = features[:training_samples]
	validation_data = features[training_samples:]

	batch_size = 32

	def take_sample(body):
		path = random.choice(body)
		with open(path) as f:
			for line in f:
				x, y = parse_csv_row_xy(line)
				sym = random.choice(featurize.symmetries)
				x, y = featurize.apply_symmetry(x, y, sym)
				return x, y
		assert False

	def take_batch(body):
		xs = np.empty((batch_size, policy_input_length))
		ys = np.empty((batch_size, policy_output_length))
		for i in range(batch_size):
			x, y = take_sample(body)
			assert len(x) == policy_input_length
			assert len(y) == policy_output_length
			xs[i, :] = x
			ys[i, :] = y
		return xs, ys

	print("Initiating training with %d (x8) training samples and %d (x8) validation samples." %
		(training_samples, validation_samples), file=sys.stderr)

	m = TFModel()
	opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	gradient_pairs = opt.compute_gradients(m.sl_policy_loss)
	gradient_tensors = [a for a, _ in gradient_pairs]
	gradient_vars = [b for _, b in gradient_pairs]
	clipped, gradient_norm = tf.clip_by_global_norm(gradient_tensors, 100)
	optimize = opt.apply_gradients(zip(gradient_tensors, gradient_vars))

	saver = tf.train.Saver()

	crnt = 0
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	if len(sys.argv) >= 3:
		saver.restore(sess, sys.argv[2])
		print("Loaded existing model from `%s`." % sys.argv[2], file=sys.stderr)

	reporter = StatisticReporter('policy-training')
	seeded = False
	while True:
		crnt += inc
		reporter.title("Epoch %d." % crnt)

		train_per_epoch = max(1, training_samples // batch_size // 2)
		val_per_epoch = max(1, validation_samples // batch_size // 2)

		for i in range(train_per_epoch):
			xs, ys = take_batch(training_data)
			_, loss, norm, acc, probs, logits = sess.run(
				[
					optimize,
					m.sl_policy_loss,
					gradient_norm,
					m.sl_policy_accuracy,
					m.policy,
					m.debug_logits],
				feed_dict={
					m.policy_input: xs,
					m.sl_policy_y_true: ys})
			if not seeded:
				random.seed(norm)
				seeded = True
			reporter.report({
				'grad_norm': norm,
				'train_loss': loss,
				'train_acc': acc}, "Epoch %d: %d/%d" % (crnt, i+1, train_per_epoch))
		for i in range(val_per_epoch):
			xs, ys = take_batch(validation_data)
			loss, acc = sess.run(
				[
					m.sl_policy_loss,
					m.sl_policy_accuracy],
				feed_dict={
					m.policy_input: xs,
					m.sl_policy_y_true: ys})
			reporter.report({
				'val_loss': loss,
				'val_acc': acc}, "Epoch %d: %d/%d" % (crnt, i+1, val_per_epoch))
		name = "%s-%d" % (filename, crnt)
		save_path = saver.save(sess, name)
		print("Model saved in file: `%s`" % save_path)
