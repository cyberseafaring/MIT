import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
env = gym.make("CarRacing-v2")
ob_data = env.observation_space.shape
print(ob_data)
wrapped_env = FlattenObservation(env)
ob_wrap = wrapped_env.observation_space.shape
print(ob_wrap)