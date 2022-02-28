import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names = ['state', 'action', 'reward', 'new_state'])

class ExperienceReplay():
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def append(self, experience):
		self.buffer.append(experience)

	def sample(self,batch_size):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)

		states, actions, rewards, is_done, new_states = zip([self.buffer[idx] for idx in indices])

		return np.array(states), np.array(actions), np.array(rewards, dtype = np.float32),np.array(is_done), np.array(new_states)

	def get_value(self):
		return self.buffer