# Cliff Walking

## Introduction 

A gridworld example from Sutton and Barto's "Reinforcement Learning" book that compares an on-policy and a off-policy method. 

## Environment 

The environment is a Cliff Walking shown in the figure below. The reward is everywhere -1 exept for the region marked as "The Cliff" where the reward is -100. 

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/cliff.png" width="150" height="150">



## Agent 

The goal of the agent is to go from the initial state to S to the final state G. Stepping into the Cliff region sends the agent instantly back to the start. 

## Methods

We use an on-policy TD method, Sarsa, and a off-policy TD method, Q-learning. 

Sarsa :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/Sarsa.png" width="150" height="150">

Q-Learning :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/Qlearning.png" width="150" height="150">

## Result 

The graphic below shows the performance of Sarsa and Q-Learning with $\epsilon$-greedy action selection, $\epsilon = 0.1$. Q-Learning learns value for the optimal policy and Sarsa learns the longer but safer path. Although Q-Learning actually learns the optimal policy, its performance is worse than that of Sarsa.

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/graph.png" width="150" height="150">



