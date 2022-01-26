import numpy as np 
import environment
import agent
from McEs import *
from environment import *
from agent import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


n_episodes = 10000000
Q = GenerateQ()
returns = np.full((2,10,2,11, n_episodes), 0, float)

for episode in range(1,n_episodes):
	
	action = np.random.randint(0,2)
	current_sum = np.random.randint(12,21)
	usable_ace = bool(np.random.randint(0,2))
	dealerCard = np.random.randint(1,11)
	player = agent.Player(current_sum,usable_ace, dealerCard)
	dealer = environment.Dealer([dealerCard])
	
	Q, returns = MonteCarlo(Q, player, dealer, action, returns, episode)

# Plot

best_policy_no_ace = np.full((10,11),0, float)
best_policy_have_ace = np.full((10,11),0, float)

for i in range(10):
	for j in range(11):
		best_policy_no_ace [i][j] = np.argmax(np.array((Q[0][i][0][j], Q[1][i][0][j])))
		best_policy_have_ace [i][j] = np.argmax(np.array((Q[0][i][1][j], Q[1][i][1][j])))

fig, ax = plt.subplots(ncols=2, figsize=(5, 5))
    
ax1, ax2 = ax

m1 = ax1.matshow(best_policy_no_ace)
m2 = ax2.matshow(best_policy_have_ace)

xticks = np.arange(1, 10)
yticks = np.arange(1, 11)

ax1.set_yticks(xticks)
ax1.set_xticks(yticks)
ax2.set_yticks(xticks)
ax2.set_xticks(yticks)

ax1.set_ylabel('Player sum', fontsize=10)
ax1.set_xlabel('Dealer showing card', fontsize=10)
ax2.set_ylabel('Player sum', fontsize=10)
ax2.set_xlabel('Dealer showing card', fontsize=10)

ax1.set_title('Policy, no usable ace', fontsize=12)
ax2.set_title('Policy, with usable ace', fontsize=12)

fig.colorbar(m1, ax=ax1)
fig.colorbar(m2, ax=ax2)

plt.show()



		

