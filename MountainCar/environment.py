import numpy as np 
from tilecode import *

class Mountain():
	def __init__(self,maxl,minl):
		self.max = maxl
		self.min = minl

	def reward(self, position):
		if position != 0.5:
			return -1
		else :
			return 0

iht = IHT(4096)

def xfunction(s,a):
	x = s[0]
	xdot = s[1]
	xi = tiles(iht, 8, [8*x / (0.5+1.2), 8*xdot/(0.07+0.07)], [a])
	return np.array(xi)

def feature(s,a):
	xi = xfunction(s,a)
	g = np.zeros(4096)
	for i in range(8):
		g[xi[i]] = 1
	return g

def qvalue(s,a,w):
	xi = xfunction(s,a)
	q = np.sum(w[xi])
	return np.array(q)
