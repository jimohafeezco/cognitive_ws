
# Cognitive Robotics


## Comparing the Modified and Original GYM environment : Exercise 4
When the half cheetah and hopper were run in the original gym environment and the modified gym environment, the result and performance were different. 
In the modififed, the hopper and halfcheetah quickly learnt how to walk on the legs after some episodes and were able to balance as against the original where after several training on different seed values, the performance were not as excellent as modified.

# Differences between the modified and the originalreward functions

## Balance Bot: Exercise 5

This is an environment for [OpenAI Gym](https://github.com/openai/gym) where the goal is to train a controller for a two-wheeled balancing robot using Q learn reinforcement learning algorithm. The aim is to stay upright as long as possible, and maintain desired speed (by default zero, ie stationary).

This work is about building from scratch a pybullet envirnoment for an inveted pendulum configuration with two wheels on the same axis. Inverted pendulum is a naturally unstable configuration, thus the aim is to keep the body of the robot upright and optionally allow movement at a desired velocity.

The robot model is specified or designed in URDF file format and saved as an XML.

In  gym environment, there are three attributes that sould be set in the env class/
 
1) action_space: The Space object corresponding to valid actions.

2) observation_space: The Space object corresponding to valid observations.

3) reward_range: A tuple corresponding to the min and max possible rewards.

This work is based on a tutorial guide in (https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/).


## Exercise 6: xdiscrim
This robot was trained with ten different seeds (1,5, 11, 14,15,16,18,20,25,30)
To train an evolutionary algorithm:
'''python3 ../bin/es.py -f ErDiscrim.ini -s <seed value>

To run a trained evolutionary algorithm
python3 ../bin/es.py -f ErDiscrim.ini -t bestgS30.npy
'''python3 ../bin/es.py -f ErDiscrim.ini -t bestgS30.npy'''
 
### Seed Performance of feedforward neural Architectures(Seed values of 11 and 30)
 
![alt text](/media/allstat.png "Plot of different seed values")

### Seed Performance of feedforward neural Architectures(Seed values of 11 and 30)
![alt text](/media/best11.gif)
![alt text](/media/best30.gif) 

### Seed Performance of feedforward neural Architectures(Seed values of 11 and 30)
![alt text](/media/best11feedforward.gif) 
![alt text](/media/best30feedforward.gif)

