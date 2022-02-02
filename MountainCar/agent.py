import numpy as np 
import random
import math
from environment import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MountainCar():

	def __init__(self,position,velocity):
		self.position = position
		self.velocity = velocity
		self.action = np.array([0,1,2])

	def bound(self, x, x_min, x_max):
		if x < x_min:
			return x_min
		else:
			return min(x, x_max)

	def move(self, a, x_bound, v_bound):
		v = self.velocity + 0.001*a - 0.0025*(math.cos(3*self.position))
		x = self.position + v
		self.position = self.bound(x, x_bound[0], x_bound[1])
		self.velocity = self.bound(v, v_bound[0], v_bound[1])
		return self.position, self.velocity
	
	def get_state(self):
		return (self.position, self.velocity)

	def reset(self, x0_min, x0_max):
		self.position = random.uniform(x0_min, x0_max)
		self.velocity = 0

	def render(self,S,file_path='/Users/Miri/RL-Practice/MountainCar/mountain_car.mp4', mode='mp4'): 
		fig = plt.figure()
		ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
		ax.grid(False)  # disable the grid
		x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
		y_sin = np.sin(3 * x_sin)
		ax.plot(x_sin, y_sin)  # plot the sine wave
		dot, = ax.plot([], [], 'ro')
		time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

		def _init():
			dot.set_data([], [])
			time_text.set_text('')
			return dot, time_text

		def animate(i):
			x = S[i]
			y = np.sin(3 * x)
			dot.set_data(x, y)
			time_text.set_text("Time: " + str(np.round(i*0.1, 1)) + "s" + '\n' + "Frame: " + str(i))
			return dot, time_text

		ani = animation.FuncAnimation(fig, animate, frames =np.arange(1,len(S)),
                                      blit=True, init_func=_init, repeat=False)


		writervideo = animation.FFMpegWriter(fps=60)
		
		ani.save(file_path, writer= writervideo) #,codec='libx264')
		fig.clear()
		plt.close(fig)
		
def Policy(s,w):
	a = np.argmax((qvalue(s,-1,w), qvalue(s,0,w), qvalue(s,1,w)))
	return a-1







