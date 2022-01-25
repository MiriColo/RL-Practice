import numpy as np 
import matplotlib.pyplot as plt 
import agent

class QTable():
	
	def __init__(self, table):
		self.table = table
		self.cliff = []
		for i in range(len(table[0])-2):
			self.cliff.append([3,i+1])
		self.length = len(table)
		self.height = len(table[0])


	def Reward(self, location):
		reward = -1
		if location in self.cliff:
			reward = -100
		if location == [self.length -1, 0]:
			reward = 0
		return reward

	def Reset(self):
		self.table = GenerateGrid()
		return self.table

def Boarder(Q):
		l,r,u,d = [],[],[],[]
		for i in range(len(Q)):
			l.append([i,0])
			r.append([i,len(Q[0])-1])
		for i in range(len(Q[0])):
			u.append(([0,i]))
			d.append([len(Q)-1, i])
		return l,r,u,d
	 		

def GenerateGrid():
	Q = np.ones((4,12))
	Q[3,11] = 0 
	return Q


def Contour(Q):
	row = np.full((1,14), -10000000000000000)
	column = np.full((4,1),-1000000000000000)
	Q_1 = np.append(Q, column, axis= 1)
	Q_2 = np.append(column, Q_1, axis =1)
	Q_3 = np.append(row, Q_2, axis =0)
	Q = np.append(Q_3, row, axis =0)

	return Q


		










