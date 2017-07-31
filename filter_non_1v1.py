import json
from pprint import pprint

with open("paths.txt") as paths:
	for path in paths:
		path = path.strip()
		with open(path) as replay_text:
			try:
				replay_str = replay_text.read()
				idx = replay_str.find('stars')
				en = replay_str.find(']', idx)
				sec = replay_str[idx:en]
				if sec.count(',') == 1:
					print path
			except:
				pass
