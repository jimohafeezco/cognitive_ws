### cognitive_ws

### Balance Bot

This is an environment for [OpenAI Gym](https://github.com/openai/gym) where the goal is to train a controller for a two-wheeled balancing robot using Q learn reinforcement learning algorithm. The aim is to stay upright as long as possible, and maintain desired speed (by default zero, ie stationary).

This work is about building from scratch a pybullet envirnoment for an inveted pendulum configuration with two wheels on the same axis. Inverted pendulum is a naturally unstable configuration, thus the aim is to keep the body of the robot upright and optionally allow movement at a desired velocity.

The robot model is specified or designed in URDF file format and saved as an XML.

In  gym environment, there are three attributes that sould be set in the env class/
 
1) action_space: The Space object corresponding to valid actions.

2) observation_space: The Space object corresponding to valid observations.

3) reward_range: A tuple corresponding to the min and max possible rewards.
This work is based on a tutorial guide in (https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/).
![alt text](/xdiscrim/best30.gif)
