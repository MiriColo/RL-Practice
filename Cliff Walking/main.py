import numpy as np 
import agent 
import Environment 
from Environment import *

n_episode = 100

for episode in range(n_episode):
	s = [3,0]
	agent1 = agent.MyAgent(s[0], s[1])
	Q = Environment.QTable(GenerateGrid())
	

	while not s == list([3,11]) or s not in Q.cliff:
		direction = Policy(Q.table,0.1,s)
		agent1.Move(direction, Q)
		s_1 = agent1.Location()
		reward = Q.Reward(s_1)
		Q.table[s[0],s[1]] = Q.table[s[0],s[1]] + 0.5*(reward + 0.9*Q.table[s_1[0],s_1[1]] - Q.table[s[0],s[1]])
		s = s_1

	
		
