import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm
import pickle5 as pickle
from tqdm.notebook import tqdm

# Create the environment
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human")
desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode="human")

# visualize the environment----------------------------------------------------------------------------------------------------------------
# env.reset()
# env.render()
# observation, action----------------------------------------------------------------------------------------------------------------
# print(env.observation_space.n, env.action_space.n)
# print("Observation space: ", env.observation_space)
# print("Observation space sample: ", env.observation_space.sample())
# print("Action space: ", env.action_space)
# print("Action space sample: ", env.action_space.sample())

# episode 10, show the environment----------------------------------------------------------------------------------------------------------------
# episodes = 10
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample() # random action, the agent should learn to take the best action.
#         observation, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         done = terminated or truncated
#         print(f"Episode: {episode}, Score: {score}")
# env.close()

# Define the Q-learning algorithm
state_space = env.observation_space.n
action_space = env.action_space.n
# print("State space: ", state_space)
# print("Action space: ", action_space)

# Initialize the Q-table
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

# Initialize the Q-table
Qtable_frozenlake = initialize_q_table(state_space, action_space)

# Define the policy
# Greedy policy
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action

# Epsilon-greedy policy
def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    else:
        action = env.action_space.sample()
    return action

# Hyperparameters
# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.0005            # Exponential decay rate for exploration prob

# Training the agent
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
  for episode in tqdm(range(n_training_episodes)):
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    # Reset the environment
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False

    # repeat
    for step in range(max_steps):
      # Choose the action At using epsilon greedy policy
      action = epsilon_greedy_policy(Qtable, state, epsilon)

      # Take action At and observe Rt+1 and St+1
      # Take the action (a) and observe the outcome state(s') and reward (r)
      new_state, reward, terminated, truncated, info = env.step(action)

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

      # If terminated or truncated finish the episode
      if terminated or truncated:
        break

      # Our next state is the new state
      state = new_state
  return Qtable

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

print(Qtable_frozenlake)

# Visualize training process
def visualize_training(Qtable, env, max_steps, n_eval_episodes, seed=None):
    """
    Visualize the training process by evaluating the agent for ``n_eval_episodes`` episodes and returns average reward, std of reward, and frames for visualization.
    :param Qtable: The Q-table
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episodes to evaluate the agent
    :param seed: The evaluation seed or seed array (for environments like taxi-v3)
    """
    episode_rewards = []
    all_frames = []  # Store frames for all episodes if needed for visualization

    for episode in tqdm(range(n_eval_episodes)):
        # Reset environment with or without seed
        if seed is not None:
            if isinstance(seed, list) and episode < len(seed):
                state, info = env.reset(seed=seed[episode])
            else:
                state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        total_rewards_ep = 0
        frames = []  # Collect frames for this episode

        for step in range(max_steps):
            # Take the action (index) that has the maximum expected future reward given that state
            action = np.argmax(Qtable[state])  # Assuming greedy_policy is a simple argmax on Qtable
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            frames.append(env.render(mode='rgb_array'))  # Consider rendering less often if memory is an issue

            if terminated or truncated:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)
        all_frames.append(frames)  # Optional: comment out if not needed to save memory

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, all_frames

# show the training process
mean_reward, std_reward, all_frames = visualize_training(Qtable_frozenlake, env, max_steps, n_eval_episodes, eval_seed)

# Save the Q-table
def save_Qtable(Qtable, path):
  with open(path, "wb") as f:
    pickle.dump(Qtable, f)
    
# Evaluate the agent
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param max_steps: Maximum number of steps per episode
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param Q: The Q-table
  :param seed: The evaluation seed array (for taxi-v3)
  """
  episode_rewards = []
  for episode in tqdm(range(n_eval_episodes)):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

# Evaluate the agent
# mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
