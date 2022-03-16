import numpy as np 
import torch
from torch import nn
from torch import optim 
from torch.distributions import MultivariateNormal
import os 
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)
DEVICE = 'cpu'

path = f'{os.path.dirname(os.path.abspath(__file__))}'

class ActorCriticNetwork(nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.shared_layers = nn.Sequential(
			nn.Conv2d(6, 6, 1, stride=1),
			nn.BatchNorm2d(6),
			nn.ReLU(inplace=False),
			nn.Conv2d(6, 6, 1, stride=1),
			nn.BatchNorm2d(6),
			nn.ReLU(inplace=False),
			nn.Conv2d(6, 6, 1, stride=1),
			nn.BatchNorm2d(6),
			)

		self.policy_layer = nn.Sequential(
			nn.Flatten(),
			nn.Linear(18,18),
			nn.ReLU(inplace=False),
			nn.Linear(18, action_size),
			nn.Softplus(),
			)

		self.value_layer = nn.Sequential(
			nn.Flatten(),
			nn.Linear(18,1),
			nn.ReLU(inplace=False),
			)

	def value(self, state):
		#state = state.astype(np.float32)
		#state = torch.from_numpy(state)
		z = self.shared_layers(state)
		value = self.value_layer(z)
			
		return value 

	def policy(self, state):
		#state = torch.from_numpy(state)
		z = self.shared_layers(state)
		policy_logits = self.policy_layer(z)
		
		return policy_logits

	def forward(self, obs):
			z = self.shared_layers(obs)
			policy_logits = self.policy_layer(z)
			value = self.value_layer(z)

			return policy_logits, value


class PPOTrainer():
	def __init__(self, env, state_size, action_size, hyperparameters):
		self.env = env 
		self._init_hyperparameters(hyperparameters)
		self.state_size = state_size
		self.action_size = action_size

		self.actor = ActorCriticNetwork(self.state_size, self.action_size)
		self.critic = ActorCriticNetwork(self.state_size, self.action_size)

		self.actor_optim = Adam(self.actor.parameters(), self.lr)
		self.critic_optim = Adam(self.critic.parameters(), self.lr)

		self.cov_var = torch.full(size=(self.action_size,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		self.logger = {
		't_so_far' : 0,     # timesteps so far
		'i_so_far' : 0,     # iterations so far
		'batch_lens': [],   # episodic lengths in batch
		'batch_rewn': [],   # episodic returns in batch
		'actor_losses': [], # losses of actor netwotk in current iteration
		} 

	def learn(self, total_timestep):
		t_so_far = 0
		i_so_far = 0

		while t_so_far < total_timestep:
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

			t_so_far += np.sum(batch_lens)
			i_so_far += 1

			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# calculate advantage 
			V, _ , _= self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) #Normalizing A_k

			for _ in range(self.n_updates_per_iteration):
				V, curr_log_probs, _ = self.evaluate(batch_obs, batch_acts)

				ratios = torch.exp(curr_log_probs - batch_log_probs)

				#calculate surrogate losses

				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1- self.clip, 1+ self.clip) * A_k

				V = torch.reshape(V, [len(V)])


				actor_loss = (-torch.min(surr1.clone(), surr2.clone())).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# calculate gradients and perform backpropagation

				self.actor_optim.zero_grad()
				torch.autograd.set_detect_anomaly(True) 
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				self.logger['actor_losses'].append(actor_loss.detach())
				torch.autograd.set_detect_anomaly(True) 



			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), path + './ppo_actor.pht')
				torch.save(self.critic.state_dict(), path + '/ppo_critic.pht' )
		return actor_loss, critic_loss

	def get_action(self, obs):
		obs = obs.astype(np.float32)
		obs = torch.from_numpy(obs)
		mean = self.actor.policy(obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		action = dist.sample()
		log_probs = dist.log_prob(action)

		return mean, log_probs.detach()


	def rollout(self):
		batch_obs = np.empty([1,6,3,1])
		batch_acts = np.empty([1,6])
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		ep_rews = []

		#t = 0 

		# Keep simulating until we've run more than or equal to specified timestep
		for t in range(self.timesteps_per_batch):
			ep_rews = []
			self.env.reset()
			done = False
			info = self.env.step(None,None)
			w1 = info['weight vector']
			#w1 = torch.from_numpy(w1)
			obs = info['next state']

			

			for ep_t in range(self.max_timesteps_per_episode):
				batch_obs = np.append(batch_obs, obs, axis=0)
				w2, log_probs = self.get_action(obs)
				w2 = w2.detach().numpy()

				info = self.env.step(w1, w2)

				obs = info['next state']
				reward = info['reward']
				contin = info['continue']
				w1 = info['weight vector']
				
				ep_rews.append(reward)
				batch_acts = np.append(batch_acts, w1, axis=0)
				batch_log_probs.append(log_probs.detach())

				if contin == 0:
					break

			batch_lens.append(ep_t+1)
			batch_rews.append(ep_rews)

		#batch_acts = batch_acts[torch.arange(batch_acts.size(0)) != 0]

		batch_acts =  np.delete(batch_acts, 0, axis=0)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_obs = np.delete(batch_obs, 0, axis=0)
		batch_obs = np.array(batch_obs, dtype=float)
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		batch_rtgs = []

		for ep_rews in reversed(batch_rews):
			discounted_reward = 0 

			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward*self.gamma
				batch_rtgs.insert(0, discounted_reward)

		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs


	def evaluate(self, batch_obs, batch_acts):

		V = self.critic.value(batch_obs)

		mean = self.actor.policy(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		return V, log_probs, mean

	def _init_hyperparameters(self, hyperparameters):
		self.timesteps_per_batch = 100                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 50           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None 

		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))












		





