import numpy as np 

class Player(object):
	def __init__(self,current_sum, usable_ace, dealersCard):
		self.current_sum = current_sum
		if usable_ace :
			self.usable_ace = 1
		elif not usable_ace :
			self.usable_ace = 0

		self.dealersCard = dealersCard
		self.using_ace = usable_ace

	def AddCards(self,card):
		if self.using_ace and self.current_sum + card > 21:
			self.using_ace = False
			self.current_sum += card - 10
		else:
			self.current_sum += card
		
	def Bust(self):
		return self.GetValue() > 21

	def GetValue(self):
		return self.current_sum

	def UpdateValue(self, update):
		self.current_sum = update
		return self.current_sum

	def GetState(self):
		return list((self.current_sum, self.usable_ace, self.dealersCard))

	def ShouldHit(self, policy):
		if policy == 1 :
			return True
		else:
			return False


def Policy(Q, state):
	state[0] -= 12
	if np.array(Q[0][state[0]][state[1]][state[2]] > Q[1][state[0]][state[1]][state[2]]):
		action = 0
	else :
		action = 1
	return action

