import gym
import numpy as np
import time
env = gym.make('CartPole-v0')
# gym.logger.set_level(40)

pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons
# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
network=10
steps = 200 
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n
# initialize the training parameters randomly by using a gaussian distribution with
# average =0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0
W1 = np.random.randn(nhiddens,ninputs,network) * pvariance # first layer
W2 = np.random.randn(noutputs, nhiddens,network) * pvariance # second layer
b1 = np.zeros(shape=(nhiddens, 1,network)) # bias first layer
b2 = np.zeros(shape=(noutputs, 1,network)) # bias second layer

# env.render()
sigma =0.02
lamda= 2
param = [[W1, b1],[ W2, b2]]
network=10
scores= np.zeros(network, dtype=float)
def training(network):
    score =0
    for i in range(network):
        observation = env.reset()
        # observation, reward, done, info = env.step(env.action_space.sample())
        # convert the observation array into a matrix with 1 column and ninputs rows
        for _ in range(steps):
            observation.resize(ninputs,1)
        # compute the netinput of the first layer of neurons
            Z1 = np.dot(W1[:, :, i], observation) + b1[:, :, i]
            # compute the activation of the first layer of neurons with the tanh function
            A1 = np.tanh(Z1)
            # compute the netinput of the second layer of neurons
            Z2 = np.dot(W2[:, :, i], A1) + b2[:, :, i]
            # compute the activation of the second layer of neurons with the tanh function
            A2 = np.tanh(Z2)
            # if actions are discrete we select the action corresponding to the most activated unit
            if (isinstance(env.action_space, gym.spaces.box.Box)):
                action = A2
            else:
                action = np.argmax(A2)
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print("reward is", reward, "for ", observation)
            scores[i] +=reward
            if done:
                break
            time.sleep(0.01)
    print(scores)
    return scores
episode=10


for e in range(episode):
        reward=training(network)
        #finding 5 best
        best = reward.argsort()[-network//2:] 
        #finding 5 worst
        worst = reward.argsort()[:network//2] 
        print("Reward during the episode {} equals:\n{}".format(e + 1, reward))

        # Updating weights and biases
        W1[:, :, worst] = W1[:, :, best] + np.random.randn(nhiddens, ninputs, network//2) * ppvariance
        W2[:, :, worst] = W2[:, :, best] + np.random.randn(noutputs, nhiddens, network//2) * ppvariance
        b1[:, :, worst] = b1[:, :, best] + np.random.randn(nhiddens, 1, network//2) * ppvariance
        b2[:, :, worst] = b2[:, :, best] + np.random.randn(noutputs, 1, network//2) * ppvariance
        env.close()
print('\nPost-evaluation in process\n')

best = scores.argsort()[-1:] 
print('The best neural networks is {}'.format(int(best)))

W1 = W1[:, :, best].reshape((nhiddens, ninputs))
W2 = W2[:, :, best].reshape((noutputs, nhiddens))
b1 = b1[:, :, best].reshape((nhiddens, 1))
b2 = b2[:, :, best].reshape((noutputs, 1))
# env.close()
# def evolution(episode):

    # return W1,W2, b1, b2

# evolution(10)
episode=10
# W1,W2, b1, b2=
for i in range(episode):
    reward_sum = 0
    observation = env.reset()
    for _ in range(steps):
        env.render()
        time.sleep(0.01)

        observation.resize(ninputs,1)
        Z1 = np.dot(W1, observation) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)

        observation, reward, done, info = env.step(action)
        reward_sum = reward_sum + reward

    print("Reward at", i + 1, "is", reward_sum)

env.close()