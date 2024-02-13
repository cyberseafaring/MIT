'''
NOOP（无操作）可以不分配特定的按键。
FIRE（加速）可以映射到空格键。
RIGHT（向右）可以映射到右箭头键。
LEFT（向左）可以映射到左箭头键。
DOWN（向下）可以映射到下箭头键。
DOWNRIGHT（向下右）可以映射到同时按下右箭头键和下箭头键。
DOWNLEFT（向下左）可以映射到同时按下左箭头键和下箭头键。
RIGHTFIRE（向右开火）和LEFTFIRE（向左开火）可以通过组合键实现，例如同时按下FIRE键（空格）和方向键。
'''
import gymnasium as gym
import pygame
import sys

# 初始化pygame
pygame.init()

# 创建游戏环境
env = gym.make("ALE/Enduro-v5", render_mode="human")
observation, info = env.reset()

# 设置动作映射
action_mapping = {
    pygame.K_SPACE: 1,  # FIRE
    pygame.K_RIGHT: 2,  # RIGHT
    pygame.K_LEFT: 3,  # LEFT
    pygame.K_DOWN: 4,  # DOWN
    # 对于组合键，如DOWNRIGHT、DOWNLEFT、RIGHTFIRE、LEFTFIRE，需要在循环中特别处理
}

# 主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 捕获键盘状态
    keys = pygame.key.get_pressed()
    action = 0  # 默认为NOOP
    if keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        action = 5  # DOWNRIGHT
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        action = 6  # DOWNLEFT
    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
        action = 7  # RIGHTFIRE
    elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
        action = 8  # LEFTFIRE
    else:
        for key, action_value in action_mapping.items():
            if keys[key]:
                action = action_value
                break
    
    # 应用动作，获取下一状态
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# 退出pygame
pygame.quit()
