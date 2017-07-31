import sys

if len(sys.argv) < 2:
	print("Please provide the target: either '1v1' or a custom game ID.", file=sys.stderr)
	sys.exit(2)

from agent import Instance, MODE_INFERENCE
import keras
from train_neural_net import blank_policy_network

m = blank_policy_network()
m.load_weights("policy")
print("Model loaded.", file=sys.stderr)

import generals
import logging
import itertools

logging.basicConfig(filename='agent-log.txt', level=logging.DEBUG)
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)

# 1v1
# g = generals.Generals('your userid', 'your username', '1v1')

# ffa
# g = generals.Generals('your userid', 'your username', 'ffa')

# 2v2 game
# g = generals.Generals('your userid', 'your username', 'team')

userid = 'redacted'
username = '[Bot] [uw] zxqfl'

def update_to_observe(update):
	me = update['player_index']
	them = 1 - me
	ret = {}
	ret['type'] = 'observe'
	ret['turn'] = update['turn']
	ret['mapWidth'] = update['cols']
	ret['mapHeight'] = update['rows']
	grid = list(itertools.chain.from_iterable(update['tile_grid']))
	for i in range(len(grid)):
		if grid[i] == -2:
			grid[i] = -3 # known mountain
		elif grid[i] == -3:
			grid[i] = -1 # unknown flat
		elif grid[i] == -4:
			grid[i] = -2 # unknown obstacle
		elif grid[i] == -1:
			grid[i] = -4 # known flat
		elif grid[i] == me:
			grid[i] = 0 # ours
		else:
			assert grid[i] == them
			grid[i] = 1 # theirs
	ret['mapOwner'] = grid
	armies = list(itertools.chain.from_iterable(update['army_grid']))
	for i in range(len(armies)):
		if grid[i] == 0 or grid[i] == 1:
			pass
		else:
			armies[i] = 0
	ret['mapForces'] = armies
	knownGenerals = []
	assert update['generals'][me][0] != -1
	def to_map_index(r, c):
		return r * update['cols'] + c
	knownGenerals.append(to_map_index(update['generals'][me][0], update['generals'][me][1]))
	if update['generals'][them][0] != -1:
		knownGenerals.append(to_map_index(update['generals'][them][0], update['generals'][them][1]))
	ret['generals'] = knownGenerals
	cities = []
	for x in update['cities']:
		cities.append(to_map_index(x[0], x[1]))
	ret['cities'] = cities + knownGenerals
	ret['forces'] = [update['armies'][me], update['armies'][them]]
	ret['land'] = [update['lands'][me], update['lands'][them]]
	return ret

def play_game(gameid):
	ai = Instance(m, MODE_INFERENCE)
	if gameid == '1v1':
		g = generals.Generals(userid, username, '1v1')
	else:
		g = generals.Generals(userid, username, 'private', gameid)

	for update in g.get_updates():
		if 'complete' in update and update['complete']:
			logging.info(str(update))
			break
		if update['turn'] >= 2000:
			break
		ai.update(update_to_observe(update))
		move = ai.predict()
		ai.state.choseMove(move)

		assert len(move) == 0 or len(move) == 2
		if len(move) == 2:
			g.moveIndices(move[0], move[1])

if __name__ == "__main__":
	gameid = sys.argv[1]
	if gameid == '1v1loop':
		while True:
			m.load_weights("policy")
			try:
				play_game('1v1')
			except AssertionError as e:
				logging.debug(str(e))
				raise
			except Exception as e:
				logging.debug(str(e))
	else:
		play_game(gameid)
