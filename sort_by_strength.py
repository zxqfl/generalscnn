import json

replays = []
with open("1v1_paths.txt") as paths:
	for path in paths:
		path = path.strip()
		try:
			with open(path) as replay_file:
				replay = json.load(replay_file)
				assert len(replay['stars']) == 2
				weakest = min(replay['stars'][0], replay['stars'][1])
				replays.append([weakest, path])
		except:
			pass

replays.sort()
replays = replays[::-1]

for r in replays:
	print("%d %s" % (r[0], r[1]))
