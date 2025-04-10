# 1. 导入一些会使用的库，numpy, torch, matplotlib, gymnasium
import gymnasium as gym
from TaskOffloadEnv import TaskOffloadEnv # 一个是文件名，一个是类名

env = TaskOffloadEnv(num_servers=3)

for episode in range(5):
    obs, info = env.reset()
    done = False

    episode_size = 10 # 每个 episode 的任务数
    count = 0

    print(f"episode{episode}:")
    while count < episode_size:
        count += 1
        action = env.action_space.sample()  # 用随机策略演示
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Info: {info}")
    print("\n")

# 2. 定义一些论文的函数1

# 3. 神经网络模型的定义


# 4. 内模型，注意一个 episode 形成一个轨迹，作为元学习的一个参数


# 5. 外模型，需要使用 pytorch 的保存神经网络参数功能，否则无法记录训练的效果


# 6. 综合任务卸载模型

# 7. 强化学习，环境，动作，状态，奖励函数