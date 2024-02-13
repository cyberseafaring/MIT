import gymnasium as gym

from stable_baselines3 import A2C # 导入A2C算法，A2C是一种基于策略梯度的算法，基本

# env = gym.make("ALE/Enduro-v5", render_mode="rgb_array")
env = gym.make("ALE/Enduro-v5", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=10000, log_interval=4)

# 交互式环境

# 渲染环境
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        obs = env.reset()

env.close()