import numpy as np 
import torch
from torch import nn
from torch import optim 
from torch.distributions.categorical import Categorical 
import seaborn as sns

sns.set()

DEVICE = 'cpu'

class ActorCriticNetwork(nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()

		self.shared_layers = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.ReLU(),
			nn.Linear(64,64),
			nn.ReLU()
			)

		self.policy_layer = nn.Sequential(
			nn.Linear(64,64),
			nn.ReLU(),
			nn.Linear(64, action_size)
			)

		self.value_layer = nn.Sequential(
			nn.Linear(64,64),
			nn.ReLU(),
			nn.Linear(64,1))

	def value(self, state):
			z = self.shared_layers(state)
			value = self.value_layer(z)
			
			return value 

	def policy(self, state):
			z = self.shared_layers(state)
			policy_logits = self.policy_layer(z)
			
			return policy_logits

	def forward(self, obs):
			z = self.shared_layers(obs)
			policy_logits = self.policy_layer(z)
			value = self.value_layer(z)

			return policy_logits, value


class PPOTrainer():
	def __init__(self, actor_critic, ppo_clip = 0.2, max_policy_train_iters=80, target_kl_div=0.01, value_train_iters=80, policy_lr=3e-4, value_lr=1e-2):
		self.ac = actor_critic
		self.ppo_clip = ppo_clip
		self.max_policy_train_iters = max_policy_train_iters
		self.target_kl_div = target_kl_div
		self.value_train_iters = value_train_iters

		value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layer.parameters())

		self.value_optim = optim.Adam(value_params, lr=value_lr)
		policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layer.parameters())

		self.policy_optim = optim.Adam(policy_params, lr=policy_lr)



	def train_policy(self, state, actions, old_log_probs, gaes):
		for _ in range(self.max_policy_train_iters):
			self.policy_optim.zero_grad()

			new_logits = self.ac.policy(state)
			new_logits = Categorical(logits=new_logits)
			new_log_probs = new_logits.log_prob(actions)

			policy_ratio = torch.exp(new_log_probs - old_log_probs)
			clipped_ratio = policy_ratio.clamp(1-self.ppo_clip, 1+ self.ppo_clip)

			clipped_loss = clipped_ratio * gaes
			full_loss = policy_ratio * gaes
			policy_loss = -torch.min(full_loss, clipped_loss).mean()

			policy_loss.backward()
			self.policy_optim.step()

			kl_div = (old_log_probs - new_log_probs).mean()
			if kl_div >= self.target_kl_div:
				break

	def train_value(self, state, returns):
		for _ in range(self.value_train_iters):
			self.value_optim.zero_grad()

			value = self.ac.value(state)
			value_loss = (returns - value) ** 2
			value_loss = value_loss.mean()

			value_loss.backward()
			self.value_optim.step()


def discount_rewards(rewards, gamma=0.99):
	new_rewards = [float(rewards[-1])]
	for i in reversed(range(len(rewards)-1)):
		new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
	
	return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
	next_values = np.concatenate([values[1:], [0]])

	deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

	gaes = [deltas[-1]]

	for i in reversed(range(len(deltas)-1)):
		gaes.append(deltas[i] + decay * gamma * gaes[-1])

	return np.array(gaes[::-1])

def rollout(model, env, max_steps=1000):
	train_data = [[], [], [], [], [] ] 
	obs = env.reset()

	ep_reward = 0

	for _ in range(max_steps):
		logits, val = model(torch.tensor(obs, dtype=torch.float32,device=DEVICE))
		act_distribution = Categorical(logits=logits)
		act = act_distribution.sample()
		act_log_prob = act_distribution.log_prob(act).item()

		act, val = act.item(), val.item()

		next_obs, reward, done, _ = env.step(act)

		for i, item in enumerate((obs, act, reward, val, act_log_prob)):
			train_data[i].append(item)

		obs = next_obs
		ep_reward += reward

		if done:
			break


	train_data = [np.asarray(x) for x in train_data]

	train_data[3] = calculate_gaes(train_data[2], train_data[3])

	return train_data, ep_reward







