# Mountain Car Task

## Introduction 

This is an example took from Sutton and Barto's book "Reinforcement Learning", chapter 10. The problem is driving an underpowered car up a steep mountain road as shown in the figure below. The difficulty is that gravity is much stronger than the car's engine. So the car has to move far away first.

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/car.png" width="174" height="141">

## Agent

The car has three possible action: full throttle forward (+1), full throttle revese (-1) and zero throttle (0). 
Its position $x_t$ and velocity $\dot{x_t}$ are updated as follow
<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/move.png" width="245" height="45">


The bound operation enforces $-1.2 < x_{t+1} < 0.5$ and $-0.07 < $\dot{x_{t+1}} < 0.07$. Each episode started from a random position $x_t \in [-0.6, -0.4)$ and zero velocity


## Method 

We implemented episodic semi-gradient Sarsa :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/sarsa.png" width="572" height="237">

## Tiles

To convert the two continous state variable to binary feature we use grid-tilings. In particular, we used the tile-coding software, avaiable at  http://incompleteideas.net/tiles/tiles3.html with iht=IHT(4096) and tiles(iht, 8, [8*x/(0.5+1.2), 8*xdot/(0.07+0.07)], A), to get the indices of the ones in the feature vector for state (x, xdot) and action A. The feature vectors $x(s,a)$ created by tile coding were then combined linearly with the parameter vector to approximate the action-value function:

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/car.png" width="228" height="32">

## Results 

In the first episode the car moves like this :

/Users/Miri/RL-Practice/MountainCar/mountain_car0.mp4 

After 100 episodes :

/Users/Miri/RL-Practice/MountainCar/mountain_car1.mp4 

And after 400 episodes:

/Users/Miri/RL-Practice/MountainCar/mountain_car3.mp4 

The figure below shows several learning curves for seme-gradient Sarsa method whith various step size

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/graph.png" width="730" height="406">



