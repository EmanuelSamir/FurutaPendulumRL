
import numpy as np
from keras.layers import Dense, Input, Lambda, Flatten
from keras.models import Model
import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


MAXEPS = 2000           # Set max number of episodes
TSTEPS = 800           # Set max number of iterations because there are undefined states
EPSILON = 0.8	
TLIM = 6
ACTIONS = [-TLIM 0 TLIM]
ANGLE_MAX_INIT = 0.4           # For state initialization


class NeuralNetwork:
	def __init__(self, input_size, output_size):
		self.input_size = input_size
		self.output_size = output_size

        optimizer = Adam(0.0002, 0.2)

		NNinput = Input(shape = self.input_size, name = 'NNinput')
		x = Dense(64, activation = 'relu')(self.input)
		x = Dense(32, activation = 'relu')(x)
		NNoutput = Dense(self.output_size, activation = 'linear')(x)
		self.model = Model(inputs = NNinput, outputs = NNoutput)	
		self.model.compile(loss = 'mean_squared_error', optimizer = optimizer)

class Environment:
	def __init__(self):
		state_init = [ 2*ANGLE_MAX_INIT * np.randn() - ANGLE_MAX_INIT, 0. ]
		self.state = state_init

	def UpdateEnv(self, action):

def PendulumDynamics(state, action):
	g = 9.8
	L = 1
	m = 1
	b = 0.01
	state = state
	# zdot = [state(2) -g/L*sin(state(1))+T]
	zdot = np.array([state[1] -g/L*sin(state[0])+T/m/L/L-b*state[1]/m/L/L])
	return zdot

def NumericalIntegration(function)


class Agent:
	def __init__(self, state_init):
		self.actions = ACTIONS
		self.state = state_init
		self.Qfuntion = NeuralNetwork(len(self.state), len(self.actions)) 
		self.actions = []


	def UpdatePolicy(self):

	def ChooseAction(self, inputs):
		Q = self.Qfuntion.predict(inputs)
		if np.random.rand(1) > EPSILON: 
			# Qhat = np.max(Q)
			Qhat_index = np.argmax(Q)
		else
			Qhat_index = np.random.randint(len(ACTIONS))
			# Qhat = 
		return Qmax, Qmax_index

def main():	
	# bestSwingUp = np.zeros(1000,1)  # Large vector, so it is replaced in the first success
	angleRange = 0.05           # Switch to second controller when q reaches the ball of radius angleRange
	rateRange = 0.1          # Switch to second controller when qd reaches the ball of radius rateRange

	# -------- Q-learning -------- #
	gamma = 0.999           # Discount rate
	epsilonDecay = 0.999   # Decay after each episode
	dt = 0.05              # Timestep of integration. Each substep lasts this long.
	success = false        

	# Initizalition

	env = Environment()
	agent = Agent(env.state)

	for step in range(TSTEPS):


	#for episode in range(MAXEPS):




def Rfunc(q, qdot):
	return -((np.pi-np.absolute(q))**1 + 0.2*(np.absolute(qdot)**2))

def wrapAngle(angle):
	normalAngle = np.absolute(2*np.pi - np.absolute(angle))  # Normalization function 
	return normalAngle

if __init__ = "__main__":
	main()


