import numpy as np
import sys
import featurize

def parse_csv_row(line):
	return np.fromstring(line, dtype=np.int32, sep=',')	

flen = featurize.FEATURE_LENGTH
n = featurize.MAXSIZE
pertile = featurize.FEATURES_PER_TILE
hlen = featurize.FEATURE_HEADER_LENGTH
channels = hlen + pertile
classes = 4

def parse_csv_row_xy(line):
	row = parse_csv_row(line)
	x_out = np.asarray(row[:-3])
	y = row[-3:]
	ynp = np.zeros((n, n, classes), np.int16)
	ynp[y[0], y[1], y[2]-1] = 1
	y_out = np.reshape(ynp, n*n*classes)
	return x_out, y_out

def parse_csv_row_xvalue(line):
	row = parse_csv_row(line)
	assert len(row) == featurize.VALUE_FEATURE_LENGTH + 1
	return row[:-1], row[-1:]

def read_input_features(l, inp=sys.stdin):
	if isinstance(inp, str):
		with open(inp, 'r') as f:
			return read_input_features(f)

	print("%d samples" % l, file=sys.stderr)
	xs = np.zeros((l, flen), np.int16)
	ys = np.zeros((l, n*n*classes), np.int16)

	i = 0
	for line in inp:
		xs[i, :], ys[i, :] = parse_csv_row_xy(line)
		i += 1
		if i % 10000 == 0:
			print("%d read from disk" % i, file=sys.stderr)
	return xs, ys
