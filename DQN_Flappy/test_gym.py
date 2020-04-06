import gym
import gym_ple
import numpy as np
env = gym.make('FlappyBird-v0')
env.reset()
for _ in range(1000):
    env.render()
    print(np.shape(env.step(env.action_space.sample())[0])) # take a random action
env.close()
