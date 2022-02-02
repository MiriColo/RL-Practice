# Mountain Car Task

## Introduction 

This is an example took from Sutton and Barto's book "Reinforcement Learning", chapter 10. The problem is driving an underpowered car up a steep mountain road as shown in the figure below. The difficulty is that gravity is much stronger than the car's engine. So the car has to move far away first.

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/car.png" width="174" height="141">

## Agent

The car has three possible action: full throttle forward (+1), full throttle revese (-1) and zero throttle (0). 
Its position x_t and velocity xdot_t are updated as follow


<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/move.png" width="245" height="45">


The bound operation enforces the position to be in the range [-1.2, 0.5] and the velocity [-0.07,0-07] . Each episode started from a random position in [0.6, 0.4)


## Method 

We implemented episodic semi-gradient Sarsa :

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/sarsa.png" width="572" height="237">

## Tiles

To convert the two continous state variable to binary feature we use grid-tilings. In particular, we used the tile-coding software, avaiable at  http://incompleteideas.net/tiles/tiles3.html with iht=IHT(4096) and tiles(iht, 8, [8*x/(0.5+1.2), 8*xdot/(0.07+0.07)], A), to get the indices of the ones in the feature vector for state (x, xdot) and action A. The feature vectors $x(s,a)$ created by tile coding were then combined linearly with the parameter vector to approximate the action-value function:

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/value.png" width="228" height="32">

## Results 

In the first episode the car moves like this :
https://user-images.githubusercontent.com/89472755/152194113-9a85dffd-7fde-4ffa-a8fd-14e3cd3558f3.mp4

After 100 episodes :

https://user-images.githubusercontent.com/89472755/152194564-9e7e4920-7d17-4d55-938d-597f1089e384.mp4


And after 400 episodes:

https://user-images.githubusercontent.com/89472755/152194676-4624d029-f5c5-4689-b534-8675e7ef9739.mp4

The figure below shows several learning curves for seme-gradient Sarsa method whith various step size

<img src="https://github.com/MiriColo/RL-Practice/blob/main/MountainCar/figure/graph.png" width="730" height="406">




