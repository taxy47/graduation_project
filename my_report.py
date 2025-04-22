# 1. 导入一些会使用的库，numpy, torch, matplotlib, gymnasium
import gymnasium as gym
from TaskOffloadEnv import TaskOffloadEnv # 一个是文件名，一个是类名

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn        # 这个是 torch 的神经网络库，torch.nn 里面有很多神经网络可以使用
import torch.optim as optim  # 这个是 torch 的优化器，torch.optim 里面有很多优化器可以使用
import torch.functional as F # 这个是 torch 的函数库，torch.nn.functional 里面有很多函数可以使用


env = TaskOffloadEnv(num_servers=3)

#TODO: 设置不同的 task，整数随机，浮点数随机，等比数列随机

num_episodes = 10 # 总共的 episode 数量， 凑成一个 task 的训练样本
for episode in range(num_episodes):
    obs, info = env.reset() # 
    done = False

    n = 20
    k = 5
    episode_size = n + k # 每个 episode 的 step 大小
    count = 0

    print(f"episode{episode}:")
    while count < episode_size:
        count += 1
        action = env.action_space.sample()  # 用随机策略演示
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Info: {info}")
        with open("episode.txt", "a+") as f:
            f.write(f"episode{episode}:") # 没有换行
            f.write(f"Action: {action}, Reward: {reward}, Info: {info}\n")
    print("\n")



# 2. 定义一些论文的函数1

# 3. 神经网络模型的定义


# 4. 内模型，注意一个 episode 形成一个轨迹，作为元学习的一个参数


# 5. 外模型，需要使用 pytorch 的保存神经网络参数功能，否则无法记录训练的效果, episode 可以存到文件中


# 6. 综合任务卸载模型

# 7. 强化学习，环境，动作，状态，奖励函数