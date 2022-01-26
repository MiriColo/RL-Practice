import numpy as np 

class Dealer(object):
	def __init__(self, cards):
		self.cards = cards

	def AddCards(self,card):
		return self.cards.append(card)

	def Bust(self):
		return self.GetValue() > 21

	def GetValue(self):

		current_sum = 0
		ace_count = 0

		for card in self.cards:
			if card == 1 :
				ace_count +=1
			else :
				current_sum += card

			for ace in range(ace_count):
				current_sum += 11
				if current_sum >21 :
					current_sum -= 10

		return current_sum

	def ShouldHit(self):
		if self.GetValue() >= 17:
			return False
		else:
			return True


def GenerateQ():
	Q = np.full((2,10,2,11), 1, float )
	Q[0][0:8][:][:] = 0
	Q[1][8:10][:][:]= 0
	return Q

def GenerateCard():
	card = np.random.randint(1, 14)
	if card > 9:
		return 10
	else:
		return card


