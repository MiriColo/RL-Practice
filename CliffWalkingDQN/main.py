import numpy as np 
import cliffwalking_env
import prova
import matplotlib.pyplot as plt
from prova import play_game

env = cliffwalking_env.CliffWalkingEnv()
gamma = 0.99
copy_step = 15
state_dim = 48
action_dim = 4
hidden_layer = 3
max_experience = 10000
min_experience = 50
batch_size = 25
lr = 1e-2
S = []
S.append([0,0])

TrainNet = prova.DQN(batch_size, gamma, min_experience, max_experience, state_dim, action_dim, hidden_layer, lr)
TargetNet = prova.DQN(batch_size, gamma, min_experience, max_experience, state_dim, action_dim, hidden_layer, lr)


n_episode = 1000
total_rewards = np.empty(n_episode)
epsilon = 0.99
decay = 0.8999
min_epsilon = 0.1
episode_array = []
step = []

for episode in range(n_episode):
	print('begin episode', episode)
	episode_array.append(episode)
	epsilon = epsilon*decay
	total_reward, losses, S, iters = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
	step.append(iters)
	total_rewards[episode] = total_reward
	print('total', total_reward)
	print(S)
	if episode % 100 == 0:
		table = env.render(S)
		fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
		m = ax.matshow(table)
		plt.show()

plt.plot(episode_array, total_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()


plt.plot(episode_array, step)
plt.xlabel('Episode')
plt.ylabel('#Step')
plt.show()