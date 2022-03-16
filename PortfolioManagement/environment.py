import pandas as pd 
import numpy as np 
from datetime import datetime
import time 
import os 
import json
import torch

path = f'{os.path.dirname(os.path.abspath(__file__))}'

eps=10e-8

def fill_zeros(x):
    return '0'*(6-len(x))+x

class Environment:

	def __init__(self, start_date, end_date, codes, features, window_length):
		self.cost = 0.0025

		data = pd.read_csv(path + '/data/America.csv',index_col=0,parse_dates=True,dtype=object)
		data["code"]=data["code"].astype(str)

		start_date = [date for date in data.index if date > pd.to_datetime(start_date)][0]
		end_date = [date for date in data.index if date < pd.to_datetime(end_date)][-1]
		#data = data[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]

		data=data.loc[data["code"].isin(codes)]
		data[features]=data[features].astype(float)

		self.M = len(codes)+1
		self.N = len(features)
		self.L = int(window_length)

		asset_dict = dict()
		datee=data.index.unique()
		date_len=len(datee)

		for asset in codes:
			asset_data = data[data["code"]==asset].reindex(datee).sort_index()
			asset_data['close'] = asset_data['close'].fillna(method='pad')
			base_price = asset_data.loc[end_date, 'close']
			asset_dict[str(asset)] = asset_data

			asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

			asset_data = asset_data.fillna(method='bfill', axis=1)

			asset_data = asset_data.drop(columns=['code'])
			asset_dict[str(asset)] = asset_data
		print(asset_data)

		self.states = []
		self.price_history = []

		t = self.L+1

		while t < date_len:
			V_close = np.ones(self.L)
			y = np.ones(1)

			for asset in codes:
				asset_data = asset_dict[str(asset)]
				V_close = np.vstack((V_close, asset_data.iloc[t-self.L-1:t-1, 0]))
				y = np.vstack((y,asset_data.iloc[t,0]/asset_data.iloc[t-1,0]))
			state = V_close
			state = state.reshape(1,self.M,self.L,self.N)
			self.states.append(state)
			self.price_history.append(y)
			t += 1

		self.reset()

	def first_ob(self):
		return self.states[self.t]

	def step(self, w1, w2):

		# w1 represents the reallocated weight at the end of time t-1
		# w2 is the allocating vector at period t-1

		if self.FLAG:
			not_terminal = 1
			#price = self.price_history[self.t]

			rew, w2,price = reward(self.price_history, w1, w2, self.t, self.cost)
			risk = 0


			self.t += 1

			if self.t == len(self.states) - 1:
				not_terminal = 0 
				self.reset()

			price = np.squeeze(price)

			info = {'reward': rew,'continue': not_terminal, 'next state': self.states[self.t], 'weight vector': w2, 'price': price,'risk':risk}

			return info
		
		else:
			print
			info = {'reward':0, 'continue': 1, 'next state': self.states[self.L + 1],
			'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),
			'price': self.price_history[self.L + 1],'risk':0}

			self.FLAG = True 

			return info

	def reset(self):
		self.t = self.L + 1
		self.FLAG = False


def reward(state, w1, w2, t, cost):
	price = state[t]
	mu = cost * np.sum(np.abs(w2[0][1:] - w1[0][1:]))
	r = (np.dot(w2,price)[0] - mu)[0]
	reward = np.log(r + eps)
	w2 = w2 / (np.dot(w2,price) + eps)

	return reward, w2, price




















