import numpy as np 
from SarsaSemiG import *
import agent
from agent import *
import environment
import matplotlib.pyplot as plt
import os


path = f'{os.path.dirname(os.path.abspath(__file__))}'

x_bound = [-1.2, 0.5]
v_bound = [-0.07,0.07]
x0_min = -0.6 
x0_max = -0.4

car_1 = agent.MountainCar(0.5,0)
car_2 = agent.MountainCar(0.5,0)
car_3 = agent.MountainCar(0.5,0)

mountain = environment.Mountain(x_bound[1], x_bound[0])

n_episode = 500
alpha_1 = 0.5/8
alpha_2 = 0.1/8
alpha_3 = 0.2/8


w_1 = np.zeros(4096)
w_2 = np.zeros(4096)
w_3 = np.zeros(4096)
T_1 = []
T_2 = []
T_3 = []
episode_array = []

for episode in range(500):
	episode_array.append(episode)
	car_1.reset(x0_min, x0_max)
	s_1 = car_1.get_state()
	a_1 = Policy(s_1,w_1)
	a_1,s_1,x_1,returns_1,t_1,w_1,X_1 = Sarsa(a_1, w_1,car_1, mountain , x_bound, v_bound, alpha_1)
	
	car_2.reset(x0_min, x0_max)
	s_2 = car_2.get_state()
	a_2 = Policy(s_2,w_2)
	a_2,s_2,x_2,returns_2,t_2,w_2,X_2 = Sarsa(a_2, w_2,car_2, mountain , x_bound, v_bound, alpha_2)
	
	car_3.reset(x0_min, x0_max)
	s_3 = car_3.get_state()
	a_3 = Policy(s_3,w_3)
	a_3,s_3,x_3,returns_3,t_3,w_3,X_3 = Sarsa(a_3, w_3,car_3, mountain , x_bound, v_bound, alpha_3)

	
	T_1.append(t_1)
	T_2.append(t_2)
	T_3.append(t_3)

	for k in range(4):
		if episode == k*100:
			file_path = os.path.join(path, 'mountain_car_'+str(k)+'.mp4')
			car_1.render(X_1, file_path )


plt.plot(episode_array, T_1)
plt.plot(episode_array,T_2)
plt.plot(episode_array,T_3)
plt.legend(['alpha = 0.5/8','alpha = 0.1/8', 'alpha = 0.2/8'])
plt.xlabel('Episode')
plt.ylabel('Steps per episode')
plt.show()













