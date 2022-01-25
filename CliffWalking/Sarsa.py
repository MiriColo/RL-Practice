import numpy as np 
import agent 
import Environment 
from Environment import *
from agent import *


def Sarsa1(s,Q, agent, total_reward = 0):
	while not s == list([3,11]) and s not in Q.cliff:
		direction = Policy(Q.table,0.1,s)
		agent.Move(direction, Q.table)
		s_1 = agent.Location()
		reward = Q.Reward(s_1)
		Q.table[s[0],s[1]] += 0.5*(reward + 0.9*Q.table[s_1[0],s_1[1]] - Q.table[s[0],s[1]])
		s = s_1
		total_reward += reward
	return total_reward

	