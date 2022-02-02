import numpy as np 
from agent import *

def Sarsa(a, w,car, mountain , x_bound, v_bound, alpha):
	s = car.get_state()
	x = s[0]
	t = 0
	returns = []
	X = []
	X.append(x)
	while x != 0.5 :

		car.move(a, x_bound, v_bound)
		s_1 = car.get_state()
		x_1 = s_1[0]
		R = mountain.reward(x_1)
		returns.append(R)

		if x_1 == 0.5:
			w += alpha*(R - qvalue(s,a,w))*feature(s,a)
			break

		a_1 = Policy(s_1,w)
		w += alpha*(R+qvalue(s_1,a_1,w)-qvalue(s,a,w))*feature(s,a)
		s = s_1
		a = a_1
		x = s[0]
		t += 1
		X.append(x)

	return a,s,x,returns,t,w,X 





