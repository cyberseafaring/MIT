# learning how to initialize the environment, observe the state, reward, and action space, and run the environment for a few steps.

import gymnasium as gym 
env = gym.make("ALE/Enduro-v5", render_mode="human") # create the environment
observation, info = env.reset() # reset the environment and get the initial observation

for _ in range(1000): # run for 1000 steps
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action) # apply the action and get the next observation, reward, terminated, truncated, and info
    #print(observation) # print the observation
    # print(reward) # print the reward

    if terminated or truncated: # if the episode is terminated or truncated
        observation, info = env.reset() # reset the environment and get the initial observation

env.close() # close the environment