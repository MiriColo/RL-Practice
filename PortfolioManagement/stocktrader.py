import json
import time
import pandas as pd
import numpy as np

class StockTrader():
	def __init__(self):
		self.reset()

	def reset(self):
		self.wealth = 10e3
		self.total_reward = 0
		self.count = 0
		self.loss = 0
		self.actor_loss = 0

		self.wealth_history = []
		self.r_history = []

	def update_summary(self, loss, actor_loss, r):
		self.loss += loss 
		self.actor_loss += actor_loss
		self.total_reward += r
		self.count += 1
		self.r_history.append(r)
		self.wealth = self.wealth * math.exp(r)
		self.wealth_history.append(self.wealth)

	def write(self, epoch):
		wealth_history = pd.Series(self.wealth_history)
		r_history = pd.Series(self.r_history)
		#w_history = pd.Series(self.w_history)

	def print_result(self,epoch,agent):
		self.total_reward = self.total_reward / 10
		print('*-----Episode: {:d}, Reward:{:.6f}%, actor_loss:{:2f}-----*'.format(epoch, self.total_reward,self.actor_loss))
		#agent.write_summary(self.loss, self.total_reward,self.actor_loss, epoch)
		#agent.save_model(epoch)
		return self.total_reward

	def plot_result(self):
		time = []
		for i in range(len(self.r_history)):
			time.append(i) 

		plt.plot(time,self.wealth_history)
		plt.show()

def parse_config(config, mode):
	codes = config["session"]["codes"]
	agent_config = config["session"]["agents"]
	features = config["session"]["features"]
	predictor, framework, window_length = agent_config
	start_date = config["session"]["start_date"]
	end_date = config["session"]["end_date"]
	market = config["session"]["market_types"]
	noise_flag, record_flag, plot_flag=config["session"]["noise_flag"],config["session"]["record_flag"],config["session"]["plot_flag"]
	reload_flag, trainable=config["session"]['reload_flag'],config["session"]['trainable']
	method=config["session"]['method']

	global epochs
	epochs = int(config["session"]["epochs"])

	if mode=='test':
		record_flag='True'
		noise_flag='False'
		plot_flag='True'
		reload_flag='True'
		trainable='False'
		method='model_free'

	print("*--------------------Training Status-------------------*")
	print('Codes:',codes)
	print("Date from",start_date,' to ',end_date)
	print('Features:',features)
	print("Agent:Noise(",noise_flag,')---Recoed(',noise_flag,')---Plot(',plot_flag,')')
	print("Market Type:",market)
	print("Predictor:",predictor,"  Framework:", framework,"  Window_length:",window_length)
	print("Epochs:",epochs)
	print("Trainable:",trainable)
	print("Reloaded Model:",reload_flag)
	print("Method",method)
	print("Noise_flag",noise_flag)
	print("Record_flag",record_flag)
	print("Plot_flag",plot_flag)

	return codes,start_date,end_date,features,agent_config,market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable