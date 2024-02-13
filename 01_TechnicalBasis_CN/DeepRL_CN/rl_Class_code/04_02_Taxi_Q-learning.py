import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm
import pickle5 as pickle
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array")

state_space = env.observation_space.n
# print("There are ", state_space, " possible states")
action_space = env.action_space.n
# print("There are ", action_space, " possible actions")

# Initialize Q-table
def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable

# Create our Q table with state_size rows and action_size columns (500x6)
Qtable_taxi = initialize_q_table(state_space, action_space)
# print(Qtable_taxi)
# print("Q-table shape: ", Qtable_taxi .shape)

# Greedy policy
def greedy_policy(Qtable, state):
  # Exploitation: take the action with the highest state, action value
  action = np.argmax(Qtable[state][:])

  return action

def epsilon_greedy_policy(Qtable, state, epsilon):
  # Randomly generate a number between 0 and 1
  random_num = random.uniform(0,1)
  # if random_num > greater than epsilon --> exploitation
  if random_num > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = greedy_policy(Qtable, state)
  # else --> exploration
  else:
    action = env.action_space.sample()

  return action

# Hyperparameters

# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.05           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
                                                          # Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

# Training
# Train the agent using Q-learning
total_rewards = []
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, total_rewards=[]):
  for episode in tqdm(range(n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state, info = env.reset()
    total_reward_episode = 0 # 初始化累积奖励为0

    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, state, epsilon)

      # Take action At and observe Rt+1 and St+1
      new_state, reward, terminated, truncated, info = env.step(action)
      total_reward_episode += reward # 累加奖励

      # Update Q(s,a)
      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

      if terminated or truncated:
        break

      # Our next state is the new state
      state = new_state

    # Append the total reward of the episode to the list
    total_rewards.append(total_reward_episode)
    
  return Qtable


Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi, total_rewards)

# print(Qtable_taxi)

# Visualize the rewards
def plot_rewards(total_rewards, title='Reward per Episode'):
    plt.figure(figsize=(12, 6))  # 设置图形的大小
    plt.plot(total_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')  # x轴标签
    plt.ylabel('Total Reward')  # y轴标签
    plt.title(title)  # 图形标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()

plot_rewards(total_rewards, title='Training Reward per Episode')





# Evaluate
def evaluate_agent(env, Qtable, n_eval_episodes, max_steps):
    total_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset(seed=eval_seed[episode] if episode < len(eval_seed) else None)
        total_rewards_episode = 0
        for step in range(max_steps):
            action = greedy_policy(Qtable, state)  # 使用贪婪策略来选择动作
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_episode += reward
            if terminated or truncated:
                break
            state = new_state
        total_rewards.append(total_rewards_episode)
    return np.mean(total_rewards), total_rewards

avg_reward, total_rewards = evaluate_agent(env, Qtable_taxi, n_eval_episodes, max_steps)
print("The average reward of the agent is: ", avg_reward)
# print("The total rewards of the agent are: ", total_rewards)

# visualize rewards with epsoide
