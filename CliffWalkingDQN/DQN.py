import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import numpy as np
from statistics import mean
import random

def tensor_state(state):
	s = np.zeros((1,48))
	s[0,state[0] + 4*state[1]] = 1 
	return s

class MyModel(tf.keras.Model):
	def __init__(self, state_dim : int, action_dim: int, num_hidden_layers:int):
		super(MyModel, self).__init__()
		self.input_layer = layers.InputLayer(input_shape=(state_dim,))
		self.hidden_layer = []
		initializer = tf.keras.initializers.HeNormal()
		
		for i in range(num_hidden_layers):
			self.hidden_layer.append(layers.Dense(25, activation='relu', kernel_initializer=initializer))

		self.output_layer = layers.Dense(action_dim, kernel_initializer=initializers.Zeros(), activation='linear')

	def call(self, inputs):
		z = self.input_layer(inputs)
		for layer in self.hidden_layer:
			z = layer(z)
		output = self.output_layer(z)

		return output

class DQN():
	def __init__(self, batch_size: int, gamma :float, min_experience:int, max_experience:int, state_dim:int, action_dim:int, num_hidden_layers:int, lr:float):
		self.optimizer = tf.optimizers.Adam(lr)
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.batch_size = batch_size
		self.gamma = gamma
		self.min_experience = min_experience
		self.max_experience = max_experience 
		self.model = MyModel(state_dim,action_dim, num_hidden_layers)
		self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

	def predict(self,inputs):
		
		return self.model.call(inputs)

	def train(self, TargetNet):
		if len(self.experience['s']) < self.min_experience:
			return 0
		
		ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
		states = np.asarray([self.experience['s'][i] for i in ids])
		#print('states', states)
		state = np.empty((0,48))
		for i in range(len(states)):
			state_tensor = tensor_state(states[i])
			state = np.append(state, state_tensor, axis=0)
		actions = np.asarray([self.experience['a'][i] for i in ids])
		rewards = np.asarray([self.experience['r'][i] for i in ids])
		#print('rewards', rewards)
		states_next = np.asarray([self.experience['s2'][i] for i in ids])
		#print('next_states', states_next)
		value = np.empty((0, 48))
		
		dones = np.asarray([self.experience['done'][i] for i in ids])
		#value_next = np.empty(len(states_next))
		for i in range(len(states_next)):
			state_next = tensor_state(states_next[i])
			value = np.append(value,state_next, axis=0)
		value_next= (np.max(TargetNet.predict(value), axis=1))
		actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
		
		with tf.GradientTape() as tape:
			#print('altro state',state.shape)
			selected_action_values = tf.math.reduce_sum(self.predict(state) * tf.one_hot(actions, self.action_dim), axis=1)
			loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
		variables = self.model.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.optimizer.apply_gradients(zip(gradients, variables))

		return loss

	def get_action(self, states, epsilon):
		if np.random.random() < epsilon:
			print('casuale')
			return np.random.choice(self.action_dim)
		else :
			print(self.predict(np.atleast_2d(states)))

			return np.argmax(self.predict(np.atleast_2d(states))[0])


	def add_experience(self, exp):
		if len(self.experience['s']) >= self.max_experience:
			for key in self.experience.keys():
				self.experience[key].pop(0)
		for key, value in exp.items():
			self.experience[key].append(value)

	def copy_weights(self, TrainNet):
		variables1 = self.model.trainable_variables
		variables2 = TrainNet.model.trainable_variables
		for v1, v2 in zip(variables1, variables2):
			v1.assign(v2.numpy())



def play_game(env, TrainNet, TargetNet, epsilon:float, copy_step:int):
	S = []
	rewards = 0
	iters = 0 
	done = False
	observations = env.reset()
	S.append(observations)
	s = tensor_state(observations)
	print('inizio', observations)
	losses = []

	while not done:
		action = TrainNet.get_action(s, epsilon)
		print('action', action)
		prev_observations = observations
		observations, reward, done = env.step(action)
		S.append(observations)
		s = tensor_state(observations)
		print('s2', observations)
		print('reward', reward)
		rewards += reward
		if done:
			reward = 0
			env.reset()
		if reward == -100.0:
			done = True

		exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done':done}
		TrainNet.add_experience(exp)


		loss = TrainNet.train(TargetNet)
		if isinstance(loss,int):
			losses.append(loss)
		else:
			losses.append(loss.numpy())
		iters += 1
		if iters % copy_step == 0:
			TargetNet.copy_weights(TrainNet)

	return rewards, mean(losses), S, iters

	





