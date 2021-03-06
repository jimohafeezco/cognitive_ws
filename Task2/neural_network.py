import gym
import numpy as np
import time
env = gym.make('CartPole-v0')
pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons
# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n
# initialize the training parameters randomly by using a gaussian distribution with
# average =0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0
W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens, 1)) # bias first layer
b2 = np.zeros(shape=(noutputs, 1)) # bias second layer

# env.render()
sigma =0.02
lamda= 2
param = [[W1, b1],[ W2, b2]]
scores =[]


def training():
    score =0
    for i_episode in range(10):
        observation = env.reset()
        # observation, reward, done, info = env.step(env.action_space.sample())
        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(ninputs,1)
        # compute the netinput of the first layer of neurons
        Z1 = np.dot(W1, observation) + b1
        # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
        # compute the netinput of the second layer of neurons
        Z2 = np.dot(W2, A1) + b2
        # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
        # if actions are discrete we select the action corresponding to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        for t in range(200):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print("reward is", reward, "for ", observation)
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
            time.sleep(0.01)
        # score +=reward
    # print(score)
    env.close()
    return score    
training()
# for i in range(10):
#     scores=training()
#     # training()
#     print(scores)
    # print(W1,b1)
    # scores.append(sum(score))
# print(scores)