# Monte Carlo 

## Introduction 

A python code for finding the optimial policy for a blackjack game, took from Sutton and Barto "Reinforcement Learning" book. The object of the game is to obtain cards whose sum is as great as possibile without exeeding 21. All face cards count as 10 and the ace can count either as 1 or as 11.

The game begins with two cards given to the agent and to the dealer. The enironement shows the agent his cards and just one card of the dealer. Each game of blackjack is an episode and at the end of the episode a reward of 1 is given if the agent wins and -1 and 0 if he lose or draw rispectivitely.  

The agent has two possible action: hit or stick. If the sum of his cards exeeds 21 we say it goes bust, in that case he loses, if he stick is the dealer's turn. The dealer stick on any sum of 17 or greater and hits otherwise. If the dealer goes bust the agent wins, otherwise the outcome is determined by whose final sum is closer to 21. 


## Method 

We implemented a Monte Carlo method with exploting starts for estimating the optimal policy.

Method:


## Result 

Here's the final policy after 100000 episode:

