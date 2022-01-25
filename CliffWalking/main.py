import numpy as np 
import agent 
import Environment 
from Environment import *
from agent import *
from Sarsa import Sarsa1
from QLearning import *


n_episode = 500
reward_array_S = []
reward_array_Q = []
episode_array = []
average_Q = []
average_S = []
s = [3,0]
agent1 = agent.MyAgent(s[0], s[1])
Q = Environment.QTable(GenerateGrid())


for episode in range(n_episode):
	s = [3,0]
	agent1.Reset()
	Q.Reset()
	episode_array.append(episode) 

	total_reward_Q = QLearning(s,Q,agent1)
	total_reward_S = Sarsa1(s,Q,agent1)

	reward_array_S.append(total_reward_S)
	average_reward_S = np.sum(reward_array_S) / np.count_nonzero(reward_array_S)
	average_S.append(average_reward_S)
	reward_array_Q.append(total_reward_Q)
	average_reward_Q = np.sum(reward_array_Q) / np.count_nonzero(reward_array_Q)
	average_Q.append(average_reward_Q)

plt.plot(episode_array, average_S)
plt.plot(episode_array, average_Q)
plt.legend(['SARSA','Q learning'])
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.show()



	
		