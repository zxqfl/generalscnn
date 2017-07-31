import random
from featurize import state_transition, value_feature

size_freqs = [
	(17, 73),
	(18, 736),
	(19, 117),
	(20, 28),
	(21, 23),
	(22, 17),
	(23, 6)]

def sample_from(dist):
	ret = None
	prob = 0
	for x, p in dist:
		prob += p
		if random.random() < p / prob:
			ret = x
	return ret

INF_DIST = 2000

MAP_UNKNOWN = -1;
MAP_UNKNOWN_OBSTACLE = -2;
MAP_MOUNTAIN = -3;
MAP_EMPTY = -4;

class GeneralsGame:
	def __init__(self):
		while True:
			self.mapWidth = sample_from(size_freqs)
			self.mapHeight = sample_from(size_freqs)
			self.totalSize = self.mapWidth * self.mapHeight
			self.mountains = set()
			candidates = set()
			for i in range(self.totalSize):
				if random.random() < 0.25:
					self.mountains.add(i)
				else:
					candidates.add(i)
			g1 = random.choice(list(candidates))
			candidates.remove(g1)
			g2 = random.choice(list(candidates))
			candidates.remove(g2)
			self.generals = [g1, g2]
			self.cities = set()
			self.armies = [0] * self.totalSize
			for x in candidates:
				if random.random() < 0.05:
					self.cities.add(x)
			self.adj = self.computeAdj()
			self.dists = self.allPairs()
			if self.dists[g1][g2] <= 12:
				continue
			if self.dists[g1][g2] >= 24:
				continue
			for x in self.generals:
				self.armies[x] = 1
			for x in self.cities:
				self.armies[x] = random.randint(40, 50)
			self.owner = [-1] * self.totalSize
			for i in range(len(self.generals)):
				self.owner[self.generals[i]] = i
			break
		self.turn = 0
		self.nextMoves = {}
		self.fogLifted = False
		self.knows = [False, False]

	def indexToPoint(self, idx):
		return idx // self.mapWidth, idx % self.mapWidth
	def pointToIndex(self, r, c):
		return r * self.mapWidth + c
	def isValidPoint(self, r, c):
		return r >= 0 and c >= 0 and r < self.mapHeight and c < self.mapWidth
	def isValidIndex(self, idx):
		return idx >= 0 and idx <= self.totalSize

	def visibleTo(self, idx, player):
		if self.fogLifted:
			return True
		r, c = self.indexToPoint(idx)
		for dr in [-1, 0, 1]:
			for dc in [-1, 0, 1]:
				nr, nc = r + dr, c + dc
				if self.isValidPoint(nr, nc):
					nidx = self.pointToIndex(nr, nc)
					if self.owner[nidx] == player:
						return True
		return False

	def computeAdj(self):
		ret = [[] for _ in range(self.totalSize)]
		for idx in range(self.totalSize):
			r, c = self.indexToPoint(idx)
			for dx, dy in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
				if self.isValidPoint(r+dx, c+dy):
					ni = self.pointToIndex(r+dx, c+dy)
					if ni not in self.mountains:
						ret[idx].append(ni)
		return ret

	def manhattanDistance(self, a, b):
		if self.isValidIndex(a) and self.isValidIndex(b):
			ar, ac = self.indexToPoint(a)
			br, bc = self.indexToPoint(b)
			return abs(ar - br) + abs(ac - bc)
		else:
			return None

	def attemptMove(self, a, b, who):
		assert who == 0 or who == 1
		assert who not in self.nextMoves
		dist = self.manhattanDistance(a, b)
		if dist == 1:
			self.nextMoves[who] = [a, b]

	def attemptMoveArray(self, move, who):
		if move == []:
			pass
		else:
			self.attemptMove(move[0], move[1], who)

	def tick(self):
		self.turn += 1
		for i in range(2):
			if i in self.nextMoves:
				move = self.nextMoves[i]
				a = move[0]
				b = move[1]
				if self.owner[a] != i:
					continue
				if self.armies[a] <= 1:
					continue
				if b in self.mountains:
					continue
				delta = self.armies[a] - 1
				self.armies[a] -= delta
				if self.owner[b] == i:
					self.armies[b] += delta
				elif delta > self.armies[b]:
					self.armies[b] = delta - self.armies[b]
					self.owner[b] = i
				else:
					self.armies[b] -= delta
		self.nextMoves = {}

		for i in range(2):
			if self.owner[self.generals[i]] != i:
				return {'winner': 1 - i}

		if self.turn % 2 == 0:
			for x in self.cities:
				if self.owner[x] >= 0:
					self.armies[x] += 1
			for x in self.generals:
				self.armies[x] += 1

		if self.turn % 50 == 0:
			for i in range(self.totalSize):
				if self.owner[i] >= 0:
					self.armies[i] += 1

		for i in range(2):
			if self.visibleTo(self.generals[1-i], i):
				self.knows[i] = True

		return {}

	def allPairs(self):
		result = []
		for src in range(self.totalSize):
			d = [None] * self.totalSize
			if src not in self.mountains:
				q = [src]
			else:
				q = []
			d[src] = 0
			nextIdx = 0
			while nextIdx < len(q):
				idx = q[nextIdx]
				nextIdx += 1
				for nidx in self.adj[idx]:
					if d[nidx] == None and nidx not in self.cities:
						d[nidx] = d[idx] + 1
						q.append(nidx)
			for i in range(len(d)):
				if d[i] == None:
					d[i] = INF_DIST
			result.append(d)
		return result

	def asObserve(self, me):
		assert me == 0 or me == 1
		them = 1 - me
		mapOwner = [None] * self.totalSize
		mapForces = [None] * self.totalSize
		for i in range(self.totalSize):
			if self.visibleTo(i, me):
				if i in self.mountains:
					mapOwner[i] = MAP_MOUNTAIN
				elif self.owner[i] == -1:
					mapOwner[i] = MAP_EMPTY
				else:
					if self.owner[i] == me:
						mapOwner[i] = 0
					else:
						assert self.owner[i] == them
						mapOwner[i] = 1
				mapForces[i] = self.armies[i]
			else:
				if i in self.mountains or i in self.cities:
					mapOwner[i] = MAP_UNKNOWN_OBSTACLE
				else:
					mapOwner[i] = MAP_UNKNOWN
				mapForces[i] = 0
		forces = [0, 0]
		land = [0, 0]
		for i in range(self.totalSize):
			if self.owner[i] >= 0:
				forces[self.owner[i]] += self.armies[i]
				land[self.owner[i]] += 1
		if me == 1:
			forces.reverse()
			land.reverse()
		cities = []
		for x in self.cities:
			if self.visibleTo(x, me):
				cities.append(x)
		generals = []
		assert self.visibleTo(self.generals[me], me), "Can't see our own general"
		generals.append(self.generals[me])
		if self.visibleTo(self.generals[them], me):
			generals.append(self.generals[them])
		for x in generals:
			cities.append(x)
		return {
			'type': 'observe',
			'turn': self.turn,
			'mapWidth': self.mapWidth,
			'mapHeight': self.mapHeight,
			'mapOwner': mapOwner,
			'mapForces': mapForces,
			'cities': cities,
			'generals': generals,
			'forces': forces,
			'land': land
		}

	def valueFeature(self):
		assert not self.fogLifted
		self.fogLifted = True
		obs = self.asObserve(0)
		self.fogLifted = False
		state = state_transition(None, obs)
		return value_feature(True, state, self.knows)

	def mapString(self):
		result = []
		for r in range(self.mapHeight):
			s = ""
			def pad(x):
				x = str(x)
				if len(x) < 4:
					x = x + " " * (4 - len(x))
				return x

			for c in range(self.mapWidth):
				idx = self.pointToIndex(r, c)
				if self.owner[idx] >= 0:
					if self.owner[idx] == 0:
						s += pad("+%d" % self.armies[idx])
					else:
						s += pad("-%d" % self.armies[idx])
				elif idx in self.mountains:
					s += pad('M')
				elif idx in self.cities:
					s += pad('C')
				else:
					s += pad('.')
			result.append(s)
		return '\n'.join(result) + '\n'
