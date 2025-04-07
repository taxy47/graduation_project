# 1. 导入一些会使用的库，numpy, torch, matplotlib, gymnasium
import gymnasium as gym
from TaskOffloadEnv import TaskOffloadEnv # 一个是文件名，一个是类名

env = TaskOffloadEnv(num_servers=3)

for episode in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 用随机策略演示
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Info: {info}")



# 2. 定义一些论文的函数1

# 3. 神经网络模型的定义


# 4. 内模型


# 5. 外模型


# 6. 综合任务卸载模型

# 7. 强化学习，环境，动作，状态，奖励函数