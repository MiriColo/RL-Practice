import numpy as np 

class MyAgent():
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def Location(self):
		return list([self.x, self.y])

	def Boarder(self,Q):
		l,r,u,d = [],[],[],[]
		for i in range(Q.length):
			l.append([i,0])
			r.append([i,Q.height-1])
		for i in range(Q.height):
			u.append(([0,i]))
			d.append([Q.length-1, i])
		return l,r,u,d


	def Move(self, direction, Q):

		l,r,u,d = self.Boarder(Q)
		
		if self.Location() not in l and direction == 0 :
			self.y -= 1
		elif self.Location() not in r and direction == 1:
			self.y += 1
		elif self.Location() not in u and direction == 2:
			self.x -=1
		elif self.Location() not in d and direction == 3 :
			self.x +=1

		return self.x, self.y






