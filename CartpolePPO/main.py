import gym 
import matplotlib.pyplot as plt
import numpy as np
import torch
import PPO2
from PPO2 import *
from PPO2 import rollout

DEVICE = 'cpu'

env = gym.make('CartPole-v0')
model = PPO2.ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n)

model = model.to(DEVICE)
train_data, reward = rollout(model, env)

n_episodes = 500
print_freq = 20

ppo = PPO2.PPOTrainer(
	model, 
	policy_lr=3e-4,
	value_lr=1e-3,
	target_kl_div=0.20,
	max_policy_train_iters=40,
	value_train_iters=40)

ep_rewards = []
episodes = []

for episode_idx in range(n_episodes):

	train_data, reward = rollout(model, env)
	episodes.append(episode_idx)
	ep_rewards.append(reward)

	permute_idxs = np.random.permutation(len(train_data[0]))

	state = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
	actions = torch.tensor(train_data[1][permute_idxs], dtype=torch.float32, device=DEVICE)
	gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
	act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)

	returns = discount_rewards(train_data[2])[permute_idxs]
	returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

	ppo.train_policy(state, actions, act_log_probs, gaes)
	ppo.train_value(state, returns)

	if (episode_idx +1) % print_freq == 0:
		print('Episode {} | Avg Reward {:.1f}'.format(
			episode_idx + 1, np.mean(ep_rewards[-print_freq:])))
		

plt.plot(episodes, ep_rewards)
plt.xlabel('episode')
plt.ylabel('Reward')
plt.show()

