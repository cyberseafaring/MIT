import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import numpy as np
from tqdm.auto import tqdm # 使用auto模块，以便在Jupyter Notebook中使用tqdm
import matplotlib.pyplot as plt
# 正常训练模式

# 创建并初始化环境
# env = gym.make("ALE/Enduro-v5", render_mode="human") # 显示图形界面
# model = DQN("MlpPolicy", env, verbose=1, buffer_size=int(1e4)) # buffer_size显著减少内存使用量。默认值为1000000，这对于大型环境来说是合理的，但对于小型环境来说浪费内存。
#buffer_size is the size of the replay buffer, which stores the experiences of the agent, and is used to sample and train the agent.

# model.learn(total_timesteps=10000)

# # 重置环境以开始新的回合
# obs = env.reset()

# for i in range(1000):
#     # 以一定概率随机选择动作，以促进探索
#     if np.random.rand() < 0.2:  # 例如，在20%的时间步中使用随机探索
#         action = env.action_space.sample()  # 从动作空间中随机选取动作
#     else:
#         action, _states = model.predict(obs, deterministic=True)

#     # 执行动作，观察新的状态和奖励
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()

#     # 打印当前步骤的动作和奖励
#     print(f"Step: {i+1}, Action: {action}, Reward: {reward}")

#     # 如果回合结束，则重置环境
#     if terminated or truncated:  # 如果回合结束或者被截断
#         obs = env.reset()
#         print("End of episode, resetting environment")

# # 关闭环境
# env.close()



# 不显示游戏界面的训练模式
# 创建环境并包装监控器以记录统计信息
env = Monitor(gym.make("ALE/Enduro-v5"))

# 初始化模型
model = DQN("MlpPolicy", env, verbose=0, buffer_size=int(1e4))

total_timesteps = 100000
log_interval = 1000  # 每1000步更新进度条和统计信息

# 准备进度条和收集奖励
with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
    episode_rewards = []  # 用于存储每个episode的总奖励
    for step in range(0, total_timesteps, log_interval):
        # 执行学习
        model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
        pbar.update(log_interval)

        # 收集奖励信息
        episode_rewards.extend(env.get_episode_rewards()[-log_interval:])

# 绘制奖励变化图
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Progress Over Episodes")
plt.legend()
plt.show()

# 关闭环境
env.close()