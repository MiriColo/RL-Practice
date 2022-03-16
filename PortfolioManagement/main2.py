import json
import time
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import environment
import ppo2
import torch
from argparse import ArgumentParser
import stocktrader
from stocktrader import parse_config


eps = 10e-8
epochs = 0 
M = 0
hyperparameters = {
				'timesteps_per_batch': 100, 
				'max_timesteps_per_episode': 50, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

total_timestep = 1000

def parse_info(info):
	return info['reward'],info['continue'],info[ 'next state'],info['weight vector'],info ['price'],info['risk']

def traversal(stocktrader, agent, env, epoch, trainable):
	info = env.step(None, None)
	r, contin, s, w1, p, risk = parse_info(info)

	loss = 0
	actor_loss = 0
	contin = 1
	l = 0
	while contin == 1 :
		l = l+1
		s = s.astype(np.float32)
		w2, _= agent.get_action(s)
		w2 = w2.detach().numpy()
		env_info = env.step(w1,w2)
		r, contin, s_next, w1, p, risk = parse_info(env_info)

		if contin == 0 :
			actor_loss, loss = agent.learn(total_timestep)
		s = s_next

		stocktrader.update_summary(loss, actor_loss, r)


def session(config, mode):
	codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable=parse_config(config,mode)
	env = environment.Environment(start_date, end_date, codes, features, window_length)
	action_size = 6
	state_size = np.array(env.states[0]).shape
	agent = ppo2.PPO(predictor,6,1,3,'-'.join(agent_config),reload_flag,trainable)
	stocktrader = stocktrader.StockTrader()
	epochs_array = []
	reward_array = []

	if mode == 'train':
		print("Training with {:d}".format(epochs))

		for epoch in range(epochs):
			epochs_array.append(epoch)
			env.reset()
			print('Now we are at epoch ', epoch)
			traversal(stocktrader, agent, env, epochs, trainable)
			reward = stocktrader.print_result(epoch, agent)
			reward_array.append(reward)
			stocktrader.write(epoch)
			if epoch % 20 == 0:
				stocktrader.plot_result()
				plt.plot(epochs_array, reward_array)
				plt.xlabel('Episode')
				plt.ylabel('Total Reward')
				plt.show()

			stocktrader.reset()
	elif mode == 'test':
		traversal(stocktrader, agent, env, epochs, trainable)
		stocktrader.write(1)
		stocktrader.plot_result()
		stocktrader.print_result(1, agent)

def build_parser():
	parser = ArgumentParser(description='Provide arguments for training PPO model in Portfolio Management')
	parser.add_argument("--mode",dest="mode",help="train, test",metavar="MODE", default="train",required=True)
	return parser

def main():
	parser = build_parser()
	args=vars(parser.parse_args())
	print(args)
	with open('config.json') as f:
		config=json.load(f)
		session(config, args['mode'])

main()










