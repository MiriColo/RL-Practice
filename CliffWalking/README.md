# Cliff Walking

## Introduction 

A gridworld example from Sutton and Barto's "Reinforcement Learning" book that compares an on-policy and a off-policy method. 

## Environment 

The environment is a Cliff Walking shown in the figure below. The reward is everywhere -1 exept for the region marked as "The Cliff" where the reward is -100. 

![image|816 × 350,20%](https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/cliff.png)

## Agent 

The goal of the agent is to go from the initial state to S to the final state G. Stepping into the Cliff region sends the agent instantly back to the start. 

## Methods

We use an on-policy TD method, Sarsa, and a off-policy TD method, Q-learning. 

Sarsa :

![image,20%](https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/Sarsa.png)

Q-Learning :

![image,20%](https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/Qlearning.png)

## Result 

The graphic below shows the performance of Sarsa and Q-Learning with $\epsilon$-greedy action selection, $\epsilon = 0.1$. Q-Learning learns value for the optimal policy and Sarsa learns the longer but safer path. Although Q-Learning actually learns the optimal policy, its performance is worse than that of Sarsa.

![image,20%](https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/graph.png)



