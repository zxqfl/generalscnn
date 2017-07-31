import sys
import time
from math import sqrt

def current_milli_time():
	return int(round(time.time() * 1000))

class StatisticReporter:
	def __init__(self, name, print_interval=10000):
		self.stats = {}
		self.name = name
		self.lastPrintTime = None
		self.changesSinceLastPrint = 0
		self.printInterval = print_interval

		self.title('Reporting statistics from `%s`.' % name)

	def print(self, s):
		print(s, file=sys.stderr)

	def title(self, t):
		self.print(t)

	def shouldPrint(self):
		return self.changesSinceLastPrint > 0 and (self.lastPrintTime is None or current_milli_time() - self.lastPrintTime > self.printInterval)

	def printStats(self, caption=None):
		self.lastPrintTime = current_milli_time()
		if caption is not None:
			self.title(caption)
		self.title('%d changes.' % self.changesSinceLastPrint)
		self.changesSinceLastPrint = 0
		names = list(self.stats)
		names.sort()
		lens = [max(6, len(s) + 1) for s in names]
		longest_time = max([len(self.stats[s]) for s in names])
		names = ['time'] + names
		lens = [6] + lens

		self.print(''.join([' ' * (l - len(name)) + name for l, name in zip(lens, names)]))
		timestep = int(sqrt(longest_time))
		while timestep <= longest_time:
			crnt = ""
			for i in range(len(names)):
				name = names[i]
				if i == 0:
					value = timestep
				elif timestep > len(self.stats[name]):
					value = None
				else:
					value = sum(self.stats[name][-timestep:]) / timestep
				if value is None:
					crnt += ' ' * lens[i]
				else:
					s = str(value)
					if len(s) >= lens[i]:
						s = s[:(lens[i]-1)]
						if abs(float(s) - value) > 0.01:
							s = str(value)
					crnt += ' ' * max(1, lens[i] - len(s)) + s
			self.print(crnt)
			timestep *= 2
		self.print('')

	def flush(self):
		if self.changesSinceLastPrint > 0:
			self.printStats()

	def report(self, entries, caption=None):
		for x in entries:
			if x not in self.stats:
				self.stats[x] = []
			self.stats[x].append(float(entries[x]))
		self.changesSinceLastPrint += 1
		if self.shouldPrint():
			self.printStats(caption)