import numpy as np 
from environment import *
from agent import *

def GenerateEpisode(Q, player, dealer, action):
	A = []
	R = []
	S = []
	T = 0
	R.append(0)

	S.append(player.GetState())

	if action == 1:
		player.AddCards(GenerateCard())
		A.append(action)
		S.append(player.GetState())
		if player.GetValue() < 21:
			policy = Policy(Q, player.GetState())
			current_sum = player.GetValue()


			while current_sum < 21 and player.ShouldHit(policy): 
				policy = Policy(Q, player.GetState())
				player.AddCards(GenerateCard())
				A.append(1)
				R.append(0)
				S.append(player.GetState())
				T += +1 
				current_sum = player.GetValue()
			

	else:
		A.append(0)

	while not dealer.Bust() and dealer.ShouldHit():
		dealer.cards.append(GenerateCard())

	if player.Bust() or player.GetValue() < dealer.GetValue() and not dealer.Bust():
		R.append(-1)
		T += 1

	elif player.GetValue() > dealer.GetValue() or dealer.Bust():
		R.append(1)
		T += 1

	elif player.GetValue() == dealer.GetValue():
		R.append(0)
		T += 1
	A.append(0)

	return A,S,T,R

def OldValue(S,A,t):
	old = np.zeros((10,4))
	for i in range(t):
		old[i] = np.array((S[i][0],S[i][1], S[i][2], A[i]))
	return old

def ActionState(A,S):
	pairs = np.zeros(4)
	for i in range(3):
		pairs[i] = S[i]
	pairs[3] = A
	return pairs


def MonteCarlo(Q, player, dealer, action, returns, episode):

	A,S,T,R = GenerateEpisode(Q, player, dealer, action)

	G = 0
	pairs = []

	for t in range(T):
		G = 0.9*G + R[t+1]
		
		if t != 0:
			old = OldValue(S,A,t)
			pairs = ActionState(A[t], S[t])

			if not any((old[:]==pairs).all(1)):
				returns[A[t]][S[t][0]-12][S[t][1]][S[t][2]][episode] = G
				x = np.array((returns[A[t]][S[t][0]-12][S[t][1]][S[t][2]][:]))
				Q[A[t]][S[t][0]-12][S[t][1]][S[t][2]] = np.sum(x) / episode
				
				
		else :
			returns[A[t]][S[t][0]-12][S[t][1]][S[t][2]][episode] = G
			x = np.array((returns[A[t]][S[t][0]-12][S[t][1]][S[t][2]][:]))
			Q[A[t]][S[t][0]-12][S[t][1]][S[t][2]] = np.sum(x)/ episode
	return Q, returns

