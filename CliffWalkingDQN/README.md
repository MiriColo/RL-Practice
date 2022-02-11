# Cliff Walking

## Introduction 

A gridworld example from Sutton and Barto's "Reinforcement Learning" book that compares an on-policy and a off-policy method. 

## Environment 

The environment is a Cliff Walking shown in the figure below. The reward is everywhere -1 exept for the region marked as "The Cliff" where the reward is -100. 

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalking/figure/cliff.png" width="408" height="175">

The goal of the agent is to go from the initial state to S to the final state G. Stepping into the Cliff region sends the agent instantly back to the start. 

## Method 

We implemented DQN method 

## Result 

After 100 episodes the path of the agent is : 

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalkingDQN/figure/100.png" width="455" height="203">

After 200 episode the agent reaches the final path :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalkingDQN/figure/200.png" width="453" height="221">

Here some graph of the total reward and the number of steps :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalkingDQN/figure/reward.png" width="643" height="481">

<img src="https://github.com/MiriColo/RL-Practice/blob/main/CliffWalkingDQN/figure/step.png" width="624" height="455">




