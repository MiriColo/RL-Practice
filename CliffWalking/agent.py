import numpy as np 
from Environment import *

class MyAgent():
	
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def Location(self):
		return list([self.x, self.y])

	def Move(self, direction, Q):

		l,r,u,d = Boarder(Q)
		
		if self.Location() not in l and direction == 0 :
			self.y -= 1
		elif self.Location() not in r and direction == 1:
			self.y += 1
		elif self.Location() not in u and direction == 2:
			self.x -=1
		elif self.Location() not in d and direction == 3 :
			self.x +=1

		return self.x, self.y

	def Reset(self):
		self.x = 3
		self.y = 0
		return self.x, self.y

def Policy(Q, epsilon, s):
	Q = Contour(Q)
	d_1 = np.argmax(np.array([Q[s[0], s[0]-1], Q[s[0], s[1]+1], Q[s[0]-1, s[1]], Q[s[0]+1, s[1]]]))
	d_2 = np.random.randint(0,4)
	direction = np.random.choice([d_1, d_2], p= [1-epsilon, epsilon])
	return direction






