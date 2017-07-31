import json
import sys
from io import StringIO
from enum import Enum
from time import perf_counter
import random
import copy
import logging
import numpy as np

MAP_UNKNOWN = -1
MAP_UNKNOWN_OBSTACLE = -2
MAP_MOUNTAIN = -3
MAP_EMPTY = -4
FEATURES_PER_STATE = 3
FEATURE_HEADER_LENGTH = 18
DIRECTION_INDICES = [19, 23, 27]
FEATURES_PER_TILE = DIRECTION_INDICES[-1] + 4

MAXSIZE = 23
INF_DIST = 1000

FEATURE_LENGTH = FEATURE_HEADER_LENGTH + FEATURES_PER_TILE * MAXSIZE * MAXSIZE

def make2d(a, b, v):
	return [[v for j in range(b)] for i in range(a)]

class Terrain(Enum):
	FLAT = 1
	MOUNTAIN = 2
	NEUTRAL_CITY = 3

class Owner(Enum):
	OURS = 1
	THEIRS = 2
	NEUTRAL = 3

class Direction(Enum):
	EAST = 1
	NORTH = 2
	WEST = 3
	SOUTH = 4

def direction_class(d):
	if d == Direction.EAST: return 1
	if d == Direction.NORTH: return 2
	if d == Direction.WEST: return 3
	if d == Direction.SOUTH: return 4
	assert False

def direction_from_class(i):
	if i == 1: return Direction.EAST
	if i == 2: return Direction.NORTH
	if i == 3: return Direction.WEST
	if i == 4: return Direction.SOUTH

class GameState:
	def __init__(self, w, h):
		self.mapWidth = w
		self.mapHeight = h
		self.cachedAllPairs = None
		self.turn = None
		self.forces = []
		self.land = []
		self.history = {}
		self.cities = set()
		self.generals = []
		self.terrain = [Terrain.MOUNTAIN] * (h * w)
		self.owner = [Owner.NEUTRAL] * (h * w)
		self.lastMoveFrom = [0] * (h * w)
		self.armies = [0] * (h * w)
		self.nextMove = None
		self.lastMove = None
		self.theyKnowOurGeneral = False

	def adj(self, idx):
		r, c = self.indexToPoint(idx)
		ret = []
		for dx, dy in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
			if self.isValidPoint(r+dx, c+dy):
				ni = self.pointToIndex(r+dx, c+dy)
				if self.terrain[ni] == Terrain.FLAT:
					ret.append(ni)
		return ret

	def theyCanSee(self, x):
		r, c = self.indexToPoint(x)
		for dr in [-1, 0, 1]:
			for dc in [-1, 0, 1]:
				nr, nc = r + dr, c + dc
				if self.isValidPoint(nr, nc) and self.owner[self.pointToIndex(nr, nc)] == Owner.THEIRS:
					return True
		return False

	def updateWithObserve(self, observe):
		assert observe['mapWidth'] == self.mapWidth
		assert observe['mapHeight'] == self.mapHeight

		o = self
		o.lastMove = o.nextMove
		if o.lastMove != None and len(o.lastMove) == 2:
			o.lastMoveFrom[o.lastMove[0]] = o.turn
		o.turn = observe['turn']
		o.forces = copy.deepcopy(observe['forces'])
		o.land = copy.deepcopy(observe['land'])
		o.history[o.turn] = {
			'forces': copy.deepcopy(o.forces),
			'land': copy.deepcopy(o.land),
		}
		for c in observe['cities']:
			o.cities.add(c)
		new_generals = observe['generals']
		for i in range(len(new_generals)):
			if i < len(o.generals):
				assert o.generals[i] == new_generals[i]
			else:
				o.generals.append(new_generals[i])
		o.nextMove = None
		new_owner = observe['mapOwner']
		new_armies = observe['mapForces']
		for i in range(len(new_owner)):
			tile = new_owner[i]
			if tile == MAP_UNKNOWN or tile == MAP_EMPTY:
				if i in o.cities and i not in o.generals:
					o.terrain[i] = Terrain.NEUTRAL_CITY
				else:
					if o.terrain[i] != Terrain.FLAT:
						o.cachedAllPairs = None
					o.terrain[i] = Terrain.FLAT
				if o.owner[i] == Owner.OURS:
					o.owner[i] = Owner.THEIRS
			if tile >= 0:
				if o.terrain[i] != Terrain.FLAT:
					o.cachedAllPairs = None
				o.terrain[i] = Terrain.FLAT
				assert tile == 0 or tile == 1, "Players should be 0 or 1 only"
				if tile == 0:
					o.owner[i] = Owner.OURS
				else:
					o.owner[i] = Owner.THEIRS
			o.armies[i] = new_armies[i]
		o.theyKnowOurGeneral = o.theyKnowOurGeneral or o.theyCanSee(o.generals[0])
		o.validate()
		return o

	def validate(self):
		assert type(self.mapWidth) is int
		assert type(self.mapHeight) is int
		assert type(self.turn) is int
		assert len(self.forces) == 2
		assert len(self.land) == 2
		for x in self.generals:
			assert x in self.cities
		if self.nextMove != None:
			assert len(self.nextMove) == 0 or len(self.nextMove) == 2
			if len(self.nextMove) == 2:
				assert type(self.nextMove[0]) is int
				assert type(self.nextMove[1]) is int

	def choseMove(self, move):
		o = self
		o.nextMove = move
		o.validate()
		return o

	def updateWithKnownMove(self, move):
		o = self
		if move['emptyMove']:
			o.nextMove = []
		else:
			o.nextMove = [move['from'], move['to']]
		o.validate()
		return o
	
	def updateWithUnknownMove(self):
		o = self
		o.nextMove = None
		return o

	def directionBetween(self, a, b):
		if b == a + 1:
			return Direction.EAST
		elif b == a - 1:
			return Direction.WEST
		elif b == a + self.mapWidth:
			return Direction.SOUTH
		elif b == a - self.mapWidth:
			return Direction.NORTH

	def inDirection(self, pos, d):
		r, c = self.indexToPoint(pos)
		if d == Direction.EAST:
			if c + 1 == self.mapWidth:
				return None
			else:
				return pos + 1
		if d == Direction.NORTH:
			if r == 0:
				return None
			else:
				return pos - self.mapWidth
		if d == Direction.WEST:
			if c == 0:
				return None
			else:
				return pos - 1
		if d == Direction.SOUTH:
			if r + 1 == self.mapHeight:
				return None
			else:
				return pos + self.mapWidth
		assert False, "Invalid direction"

	def indexToPoint(self, idx):
		return idx // self.mapWidth, idx % self.mapWidth
	def pointToIndex(self, r, c):
		return r * self.mapWidth + c
	def isValidPoint(self, r, c):
		return r >= 0 and c >= 0 and r < self.mapHeight and c < self.mapWidth

	def lowerHalf(self, idx):
		r, c = self.indexToPoint(idx)
		return r * 2 >= self.mapHeight

	def rightHalf(self, idx):
		r, c = self.indexToPoint(idx)
		return c * 2 >= self.mapWidth

	def rotate90(self):
		assert self.mapWidth == self.mapHeight
		def f(idx):
			r, c = self.indexToPoint(idx)
			return self.pointToIndex(c, self.mapWidth - r - 1)
		return [f(i) for i in range(self.mapWidth * self.mapHeight)]

	def flipHorizontal(self):
		def f(idx):
			r, c = self.indexToPoint(idx)
			return self.pointToIndex(r, self.mapWidth - c - 1)
		return [f(i) for i in range(self.mapWidth * self.mapHeight)]

	def flipVertical(self):
		def f(idx):
			r, c = self.indexToPoint(idx)
			return self.pointToIndex(self.mapHeight - r - 1, c)
		return [f(i) for i in range(self.mapWidth * self.mapHeight)]

	def allPairs(self):
		if self.cachedAllPairs != None:
			return self.cachedAllPairs
		result = []
		for src in range(len(self.owner)):
			d = [None] * len(self.owner)
			if self.terrain[src] == Terrain.FLAT:
				q = [src]
			else:
				q = []
			d[src] = 0
			nextIdx = 0
			dx = [-1, 0, 1, 0]
			dy = [0, -1, 0, 1]
			while nextIdx < len(q):
				idx = q[nextIdx]
				r, c = self.indexToPoint(idx)
				nextIdx += 1
				for i in range(len(dx)):
					nr = r + dx[i]
					nc = c + dy[i]
					if self.isValidPoint(nr, nc):
						nidx = self.pointToIndex(nr, nc)
						if self.terrain[nidx] == Terrain.FLAT and d[nidx] == None:
							d[nidx] = d[idx] + 1
							q.append(nidx)
			for i in range(len(d)):
				if d[i] == None:
					d[i] = INF_DIST
			result.append(d)
		self.cachedAllPairs = result
		return result

	def closestThreateningEnemyDist(self):
		result = 50
		dists = self.allPairs()
		for i in range(len(self.owner)):
			if self.owner[i] == Owner.THEIRS:
				if self.armies[i] > self.armies[self.generals[0]]:
					result = min(result, dists[self.generals[0]][i])
		return result

	def enemyDist(self):
		enemies = []
		for i in range(len(self.owner)):
			if self.owner[i] == Owner.THEIRS:
				enemies.append(i)
		dists = self.allPairs()
		ret = []
		for i in range(len(self.owner)):
			d = 1000
			for o in enemies:
				if dists[i][o] != INF_DIST:
					d = min(d, dists[i][o])
			if self.terrain[i] != Terrain.FLAT:
				assert d == 1000, "terrain[%d] = %s but enemyDist[%d] = %d" % (i, str(self.terrain[i]), i, d)
			if d == 1000:
				ret.append(INF_DIST)
			else:
				ret.append(d)
		return ret

	def print(self, f=sys.stdin):
		def pad(x):
			return x + " " * (4 - len(x))
		for r in range(self.mapHeight):
			s = ""
			for c in range(self.mapWidth):
				idx = r * self.mapWidth + c
				if self.terrain[idx] == Terrain.MOUNTAIN:
					s += pad('M')
				elif self.terrain[idx] == Terrain.NEUTRAL_CITY:
					s += pad('C')
				elif self.owner[idx] == Owner.NEUTRAL:
					s += pad('.')
				elif self.owner[idx] == Owner.OURS:
					s += pad(str(self.armies[idx]))
				else:
					s += pad(str(-self.armies[idx]))
			print(s, file=f)
		print("move=%s" % str(self.nextMove), file=f)
		print("", file=f)

	def moveToY(self, move):
		d = self.directionBetween(move[0], move[1])
		r, c = self.indexToPoint(move[0])
		result = np.zeros(MAXSIZE * MAXSIZE * 4)
		result[r*MAXSIZE*4 + c*4 + (direction_class(d)-1)]
		return result

class Predictor:
	def __init__(self, state):
		self.state = state
		self.tiles = []
		f = features_from_tile(state)
		self.rows = [f]

	def xs(self):
		return self.rows

	def interpret(self, ys, mode):
		from agent import MODE_INFERENCE, MODE_TRAINING
		from simulator import sample_from

		if self.state.turn < 24:
			return []
		assert len(ys) == MAXSIZE * MAXSIZE * 4
		pos = None
		direction = None
		# conf = 0
		conf = -1
		state = self.state
		enemyDist = state.enemyDist()
		dist = state.allPairs()
		considerCount = 0
		allPoss = []

		for i in range(len(state.owner)):
			row, col = state.indexToPoint(i)
			if state.owner[i] != Owner.OURS or state.armies[i] <= 1:
				continue
			for d in range(4):
				idx = row * MAXSIZE + col
				cdir = direction_from_class(d + 1)
				j = state.inDirection(i, cdir)
				if j == None or state.terrain[j] == Terrain.MOUNTAIN:
					continue
				c = ys[idx*4+d]
				if len(state.generals) >= 2 and j == state.generals[1] and state.armies[j] < state.armies[i]:
					c = 100
				allPoss.append(((i, direction_from_class(d+1)), c))
				considerCount += 1
				if c > conf:
					conf = c
					pos = i
					direction = direction_from_class(d+1)
		
		if len(allPoss) >= 1 and mode == MODE_TRAINING:
			pos, direction = sample_from(allPoss)

		if pos == None:
			mr, mc = -1, -1
		else:
			mr, mc = state.indexToPoint(pos)
		logging.info("Turn {}. Chose move [row={}, col={}, {}] with confidence {} after considering {} moves.".format(state.turn // 2, mr, mc, direction, conf, considerCount))
		if pos == None:
			return []
		else:
			o = self.state.inDirection(pos, direction)
			assert self.state.directionBetween(pos, o) == direction
			return [pos, o]

def compose(p1, p2):
	assert len(p1) == len(p2)
	ret = [0] * len(p1)
	for i in range(len(p1)):
		ret[i] = p2[p1[i]]
	return ret

def isGroup(g, f):
	for a in g:
		for b in g:
			if f(a, b) not in g:
				return False
	# for a in g:
	# 	for b in g:
	# 		for c in g:
	# 			if f(f(a, b), c) != f(a, f(b, c)):
	# 				return False
	identity = None
	for a in g:
		isId = True
		for b in g:
			if f(a, b) != b:
				isId = False
				break
		if isId:
			identity = a
			break
	if identity == None:
		return False

	for a in g:
		hasInv = False
		for b in g:
			if f(a, b) == identity:
				hasInv = True
				break
		if not hasInv:
			return False
	return True

def subsets(xs):
	if xs == []:
		return [[]]
	else:
		o = subsets(xs[1:])
		return o + [[xs[0]] + y for y in o]

def findSymmetryPermutations():
	dummy = GameState(MAXSIZE, MAXSIZE)
	tiles = [dummy.rotate90(), dummy.flipHorizontal(), dummy.flipVertical()]
	result_tiles = []
	result_dirs = []
	for ss in subsets([i for i in range(len(tiles))]):
		p = [i for i in range(len(dummy.owner))]
		for x in ss:
			p = compose(p, tiles[x])
		result_tiles.append(p)
		r = MAXSIZE // 2
		c = MAXSIZE // 2
		idx = dummy.pointToIndex(r, c)
		pdir = [1, 2, 3, 4]
		for i in range(len(pdir)):
			d = direction_from_class(pdir[i])
			pdir[i] = direction_class(dummy.directionBetween(p[idx], p[dummy.inDirection(idx, d)]))
			pdir[i] -= 1
		result_dirs.append(pdir)
	return result_tiles, result_dirs

def findSymmetries():
	ptile, pdir = findSymmetryPermutations()
	assert isGroup(ptile, compose)
	assert isGroup(pdir, compose)
	result = []
	for i in range(len(ptile)):
		def f(x):
			if x < FEATURE_HEADER_LENGTH:
				return x
			ti = (x - FEATURE_HEADER_LENGTH) // FEATURES_PER_TILE
			pti = ptile[i][ti]
			si = (x - FEATURE_HEADER_LENGTH) %  FEATURES_PER_TILE
			psi = si
			for di in DIRECTION_INDICES:
				if di <= si and si < di + 4:
					psi = si + pdir[i][si-di] - (si-di)
			x += (pti - ti) * FEATURES_PER_TILE + (psi - si)
			return x
		px = [f(i) for i in range(FEATURE_LENGTH)]
		def fy(y):
			ti = y // 4
			si = y % 4
			pti = ptile[i][ti]
			psi = pdir[i][si]
			return y + (pti - ti) * 4 + (psi - si)
		py = [fy(i) for i in range(MAXSIZE * MAXSIZE * 4)]
		tup = (px, py)
		result.append(tup)
	return result

cachedSymmetries = None
NUM_SYMMETRIES = 8
symmetries = [i for i in range(NUM_SYMMETRIES)]
inverse_symmetry = [0, 1, 2, 3, 7, 5, 6, 4]

def apply_symmetry(x_in, y_in, sym):
	global cachedSymmetries
	if cachedSymmetries == None:
		cachedSymmetries = findSymmetries()
		for i in symmetries:
			assert compose(cachedSymmetries[i][0], cachedSymmetries[inverse_symmetry[i]][0]) == cachedSymmetries[0][0]
		# 	for j in symmetries:
		# 		if compose(cachedSymmetries[i][0], cachedSymmetries[j][0]) == cachedSymmetries[0][0]:
		# 			print("Inverse of", i, "is", j)
	assert x_in is None or len(x_in) == FEATURE_LENGTH, "Expected len(x)=%d, found %d" % (FEATURE_LENGTH, len(x_in))
	assert y_in is None or len(y_in) == MAXSIZE * MAXSIZE * 4
	px, py = cachedSymmetries[sym]

	if x_in is None:
		x_out = None
	else:
		x_out = np.copy(x_in)
		for i in range(len(x_in)):
			x_out[px[i]] = x_in[i]

	if y_in is None:
		y_out = None
	else:
		y_out = np.copy(y_in)
		for i in range(len(y_in)):
			y_out[py[i]] = y_in[i]

	return x_out, y_out

def direction_delta(r, c, d):
	if d == Direction.EAST:
		return r, c + 1
	elif d == Direction.NORTH:
		return r - 1, c
	elif d == Direction.WEST:
		return r, c - 1
	elif d == Direction.SOUTH:
		return r + 1, c

def features_from_tile(state):
	dists = state.allPairs()
	enemyDist = state.enemyDist()

	f = []
	f.append(state.turn)
	f.append(state.turn % 50)
	f.append(int(enemyDist[state.generals[0]] != INF_DIST))
	f.append(state.closestThreateningEnemyDist())
	f.append(int(len(state.generals) >= 2))
	f.append(int(state.theyKnowOurGeneral))
	for dt in [0, -1, -2]:
		t = state.turn + dt
		if t in state.history:
			entry = state.history[t]
			f.append(entry['forces'][0])
			f.append(entry['forces'][1])
			f.append(entry['land'][0])
			f.append(entry['land'][1])
		else:
			f.extend([0] * 4)

	assert len(f) == FEATURE_HEADER_LENGTH, "Wrong header length"
	for r in range(MAXSIZE):
		for c in range(MAXSIZE):
			crnt = []
			dirIndices = []
			if not state.isValidPoint(r, c):
				crnt = [0] * FEATURES_PER_TILE
			elif state.terrain[state.pointToIndex(r, c)] == Terrain.MOUNTAIN:
				crnt = [0] * FEATURES_PER_TILE
			else:
				x = state.pointToIndex(r, c)
				
				army = []
				army.append(1)
				army.append(int(state.armies[x] == 1))
				army.append(int(state.armies[x] == 2))
				army.append(int(state.armies[x] == 3))
				army.append(int(4 <= state.armies[x] <= 20))
				army.append(state.armies[x])

				if state.owner[x] == Owner.OURS:
					crnt += army
					crnt += [0] * len(army)
					crnt += [0]
				elif state.owner[x] == Owner.THEIRS:
					crnt += [0] * len(army)
					crnt += army
					crnt += [0]
				else:
					assert state.owner[x] == Owner.NEUTRAL
					crnt += [0, 0] * len(army)
					crnt += [state.armies[x]]

				crnt.append(int(state.terrain[x] == Terrain.FLAT))
				crnt.append(int(x in state.cities and x not in state.generals))

				crnt.append(int(x == state.generals[0]))
				crnt.append(int(len(state.generals) >= 2 and x == state.generals[1]))

				crnt.append(normDist(dists[state.generals[0]][x]))
				if len(state.generals) >= 2:
					crnt.append(normDist(dists[state.generals[1]][x]))
				else:
					crnt.append(normDist(INF_DIST))

				def onDirections(cb):
					dirIndices.append(len(crnt))
					for d in [Direction.EAST, Direction.NORTH, Direction.WEST, Direction.SOUTH]:
						nr, nc = direction_delta(r, c, d)
						if state.isValidPoint(nr, nc):
							idx = state.pointToIndex(nr, nc)
							if state.terrain[idx] == Terrain.FLAT:
								crnt.append(int(cb(x, idx)))
							else:
								crnt.append(0)
						else:
							crnt.append(0)

				onDirections(lambda x, y: dists[state.generals[0]][y] < dists[state.generals[0]][x])
				if len(state.generals) >= 2:
					onDirections(lambda x, y: dists[state.generals[1]][y] < dists[state.generals[1]][x])
				else:
					onDirections(lambda x, y: 0)
				onDirections(lambda x, y: enemyDist[y] < enemyDist[x])

				assert dirIndices == DIRECTION_INDICES, "Wrong direction indices (expected {}, found {})".format(DIRECTION_INDICES, dirIndices)
			assert len(crnt) == FEATURES_PER_TILE, "Wrong features per tile (expected %d, found %d)" % (FEATURES_PER_TILE, len(crnt))
			f.extend(crnt)
	return f

def split_feature(f):
	idx = 0
	header = f[idx:FEATURE_HEADER_LENGTH]
	idx += FEATURE_HEADER_LENGTH
	tiles = f[idx:idx + MAXSIZE*MAXSIZE*FEATURES_PER_TILE]
	idx += MAXSIZE*MAXSIZE*FEATURES_PER_TILE
	y = f[idx:]
	return header, tiles, y

def explain_feature(f):
	header, tiles, y = split_feature(f)
	print("Turn %d." % header[0], file=sys.stderr)
	print("We have %d troops." % header[2], file=sys.stderr)
	print("They have %d troops." % header[3], file=sys.stderr)
	print("We have %d squares." % header[4], file=sys.stderr)
	print("They have %d squares." % header[5], file=sys.stderr)
	print("Found enemy: %s." % ['no', 'yes'][header[6]], file=sys.stderr)
	print("Thread distance %d." % header[7], file=sys.stderr)
	def printTiles(func):
		def pad(s):
			return s + " " * (4 - len(s))
		idx = 0
		for r in range(MAXSIZE):
			s = ""
			for c in range(MAXSIZE):
				tile = tiles[idx:idx+FEATURES_PER_TILE]
				idx += FEATURES_PER_TILE
				if tile[14] == 0:
					s += pad('M')
				else:
					s += pad(str(func(tile)))
			print(s, file=sys.stderr)

	def ttype(tile):
		if tile[15]:
			return 'C'
		elif tile[14] == 0:
			return 'M'
		elif tile[0]:
			return '+' + str(tile[6])
		elif tile[7]:
			return '-' + str(tile[13])
		else:
			return '.'
	print("The map looks like this:", file=sys.stderr)
	printTiles(ttype)
	print("East goes to our general:", file=sys.stderr)
	printTiles(lambda x: x[22])
	print("North goes to our general:", file=sys.stderr)
	printTiles(lambda x: x[23])
	print("West goes to our general:", file=sys.stderr)
	printTiles(lambda x: x[24])
	print("South goes to our general:", file=sys.stderr)
	printTiles(lambda x: x[25])


	if len(y) == 0:
		print("There is no label for this feature.", file=sys.stderr)
	elif len(y) == 3:
		r, c = y[0], y[1]
		print("The label is to start at (%d, %d) and move %s." % (r, c, [None, "east", "north", "west", "south"][y[2]]), file=sys.stderr)
	else:
		raise Exception("Wrong feature length. Expected len(y)=3, found len(y)=%d" % len(y))
	print("", file=sys.stderr)

def print_csv_row(a, to):
	for x in a:
		assert isinstance(x, int)
	print(",".join([str(x) for x in a]), file=to)

def training_features(state):
	if state.nextMove != None and len(state.nextMove) >= 2:
		f = features_from_tile(state)
		r, c = state.indexToPoint(state.nextMove[0])
		direction = state.directionBetween(state.nextMove[0], state.nextMove[1])
		cl = direction_class(direction)
		f.append(r)
		f.append(c)
		f.append(cl)
		return [f]
	else:
		return []
	# poss = set()
	# for i in range(len(state.owner)):
	# 	if state.owner[i] == Owner.OURS and state.armies[i] >= 2:
	# 		poss.add(i)
	# used = []
	# move = [-1, -1]
	# if state.nextMove != None and len(state.nextMove) == 2:
	# 	assert state.nextMove[0] in poss, "Illegal move start location"
	# 	poss.remove(state.nextMove[0])
	# 	used.append(state.nextMove[0])
	# 	move = state.nextMove
	# list_poss = list(poss)
	# random.shuffle(list_poss)
	# used += list_poss[:min(len(list_poss), FEATURES_PER_STATE)]
	# result = []
	# for x in used:
	# 	t, p = features_from_tile(state, x)
	# 	if x == move[0]:
	# 		direction = state.directionBetween(p[move[0]], p[move[1]])
	# 		c = direction_class(direction)
	# 		result.append(t + [c])
	# 	else:
	# 		result.append(t + [0])
	return result

def state_transition(state, observe):
	if state == None:
		state = GameState(observe['mapWidth'], observe['mapHeight'])
	state = state.updateWithObserve(observe)
	return state

def all_states(eseq):
	state = None
	for i in range(len(eseq)):
		event = eseq[i]
		if event['type'] == 'observe':
			state = state_transition(state, event)
			if i + 1 < len(eseq) and eseq[i+1]['type'] == 'move':
				state = state.updateWithKnownMove(eseq[i+1])
			yield state

def normDist(x):
	if x == INF_DIST:
		return 50
	else:
		return x

VALUE_FEATURE_HEADER_LENGTH = 8
VALUE_FEATURES_PER_TILE = 18
VALUE_FEATURE_LENGTH = VALUE_FEATURE_HEADER_LENGTH + VALUE_FEATURES_PER_TILE * MAXSIZE * MAXSIZE

def value_feature(won, state, knows):
	assert len(knows) == 2 and isinstance(knows[0], bool)
	f = []
	f.append(state.turn)
	f.append(state.turn % 50)
	f.append(state.forces[0])
	f.append(state.forces[1])
	f.append(state.land[0])
	f.append(state.land[1])
	f.append(int(knows[0]))
	f.append(int(knows[1]))
	assert len(f) == VALUE_FEATURE_HEADER_LENGTH, "Wrong value header length"
	dists = state.allPairs()
	for r in range(MAXSIZE):
		for c in range(MAXSIZE):
			crnt = []
			dirIndices = []
			if not state.isValidPoint(r, c):
				crnt = [0] * VALUE_FEATURES_PER_TILE
			else:
				x = state.pointToIndex(r, c)
				
				army = []
				army.append(1)
				army.append(int(state.armies[x] == 1))
				army.append(int(state.armies[x] == 2))
				army.append(int(3 <= state.armies[x] <= 40))
				army.append(int(41 <= state.armies[x]))
				army.append(state.armies[x])

				if state.owner[x] == Owner.OURS:
					crnt += army
					crnt += [0] * len(army)
				elif state.owner[x] == Owner.THEIRS:
					crnt += [0] * len(army)
					crnt += army
				else:
					assert state.owner[x] == Owner.NEUTRAL
					crnt += [0, 0] * len(army)

				crnt.append(int(state.terrain[x] == Terrain.FLAT))
				crnt.append(int(x in state.cities and x not in state.generals))

				crnt.append(int(x == state.generals[0]))
				crnt.append(int(x == state.generals[1]))

				crnt.append(normDist(dists[state.generals[0]][x]))
				crnt.append(normDist(dists[state.generals[1]][x]))

			assert len(crnt) == VALUE_FEATURES_PER_TILE, "Wrong features per tile (expected %d, found %d)" % (FEATURES_PER_TILE, len(crnt))
			f.extend(crnt)
	if won:
		f.append(1)
	else:
		f.append(-1)
	return f

def process_eseq_value(eseq):
	result = []
	state = None
	for event in eseq:
		if event['type'] == 'observe':
			state = state_transition(state, event)
			result.append(value_feature(eseq[0]['won'], state))
	return result

def process_replay(tup):
	game_string, isValue = tup
	game = json.load(StringIO(game_string))
	assert game[0]['type'] == 'metadata'
	metadata = game[0]
	result = []
	try:
		if isValue:
			result = process_eseq_value(game)
		else:
			assert metadata['won'] == True, "Input for policy features should always be from winning side"
			for state in all_states(game):
				result.extend(training_features(state))
	except Exception as e:
		print("Warning: encountered error `%s` while procssing replay %s; skipping" % (str(e), metadata['replay_id']), file=sys.stderr)
	return result

if __name__ == "__main__":
	from multiprocessing import Pool
	import os
	target_dir = sys.argv[1]
	print("Saving all features to `%s`" % target_dir, file=sys.stderr)
	assert os.path.isdir(target_dir)
	target_dir = os.path.abspath(target_dir)
	for file in os.listdir(target_dir):
		if file.startswith("feature-"):
			print("It seems that this directory already contains features.", file=sys.stderr)
			print("Please delete them manually and rerun the script.", file=sys.stderr)
			sys.exit(2)

	isValue = False
	if len(sys.argv) >= 3:
		valstr = sys.argv[2]
		assert valstr == 'value' or valstr == 'policy'
		isValue = (valstr == 'value')

	p = Pool(8)

	cnt = 0
	def process_row(row):
		global cnt
		cnt += 1
		name = os.path.join(target_dir, "feature-%d.csv" % cnt)
		print(name)
		with open(name, 'w') as f:
			print_csv_row(row, f)

	crnt = []
	def spill():
		global crnt
		for rows in p.map(process_replay, crnt):
			for row in rows:
				process_row(row)
		crnt = []

	for line in sys.stdin:
		crnt.append((line, isValue))
		if len(crnt) == 32:
			spill()
	spill()

	print("Wrote %d features." % cnt, file=sys.stderr)
