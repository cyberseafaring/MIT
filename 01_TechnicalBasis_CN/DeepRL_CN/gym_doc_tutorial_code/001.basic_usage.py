from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

env = gym.make('LunarLander-v2', render_mode='human')
print(env.observation_space)
print("-----------------")
wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space)