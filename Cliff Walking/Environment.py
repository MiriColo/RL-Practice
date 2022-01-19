import numpy as np 
import matplotlib.pyplot as plt 
import agent

class QTable():
	def __init__(self, table):
		self.table = table
		self.cliff = []
		for i in range(len(table)-2):
			self.cliff.append([i+1,0])
		self.length = len(table)
		self.height = len(table[0])


	def Reward(self, location):
		reward = -1
		if location in self.cliff:
			reward = -100
		if location == [self.length -1, 0]:
			reward = 0
		return reward
	 		

def GenerateGrid():
	Q = np.ones((4,12))
	Q[3,11] = 0 
	return Q

def Policy(Q, epsilon, s):
	Q = Contour(Q)
	d_1 = np.argmax(np.array([Q[s[0], s[0]-1], Q[s[0], s[1]+1], Q[s[0]-1, s[1]], Q[s[0]+1, s[1]]]))
	d_2 = np.random.randint(0,4)
	direction = np.random.choice([d_1, d_2], p= [1-epsilon, epsilon])
	return direction


def Contour(Q):
	row = np.full((1,14), -1000000)
	column = np.full((4,1),-1000000)
	Q_1 = np.append(Q, column, axis= 1)
	Q_2 = np.append(column, Q_1, axis =1)
	Q_3 = np.append(row, Q_2, axis =0)
	Q = np.append(Q_3, row, axis =0)

	return Q


		










