import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class cloud_server(): # 云服务器
    pass

class edge_server(): # 边缘服务器
    pass

class Device(): # 设备
    pass

class Task(): # 任务
    pass

class Workflow(): # 工作流
    pass

class TaskOffloadEnv(gym.Env):
    def __init__(self, num_servers=3):
        super(TaskOffloadEnv, self).__init__()
        self.num_servers = num_servers

        # 任务状态空间：每个任务有 v_i 任务大小, e_i 任务间隔大小, 那么是否还要进行 action mapping
        self.observation_space = spaces.Box(low=2.0, high=5.0, shape=(2,), dtype=np.float32)

        # 动作空间：选择将任务分配到哪个边缘服务器, 0: 本地计算, 1: 边缘服务器, 2: 云服务器，云服务器和本地是否直连
        self.action_space = spaces.Discrete(num_servers)

        # 边缘服务器特性（计算能力和网络延迟), 其实需要计算得知而不是随机生成
        self.server_compute_power = np.random.uniform(5, 10, size=(num_servers,))
        self.server_latency = np.random.uniform(0.05, 0.2, size=(num_servers,))

        self.cloud_C = 150 # 150 MHz
        self.edge_C = 70 # 70 MHz
        self.local_C = 30 # 30 MHz 

        # Todo: 放在实现类中，并且使用字典或者列表来存储
        self.B_0_1 = 800 # 800 MBps
        self.B_1_2 = 200 # 200 MBps
        self.B_0_2 = 10 # 10 MBps 云服务器传输很慢，计算很强

        self.d_cloud = 0.1 # J/MB， 云服务器单位 byte 消耗少
        self.d_edge = 0.15 # J/MB
        self.d_local = 0.3 # J/MB

        self.alpha = 1 
        self.beta = 1

        self.delta = 0.3

    def reset(self, seed=None, options=None): # 外部会初始时调用
        super().reset(seed=seed)
        self.current_task = self._generate_task()
        return self.current_task, {}

    def _generate_task(self): 
        cpu = np.random.uniform(1, 10)
        deadline = np.random.uniform(0.1, 1.0)
        # return np.array([cpu / 10.0, deadline], dtype=np.float32)
        return self.observation_space.sample() # 直接使用 gym 的 sample 方法生成随机数

    def step(self, action):
        cpu_need = self.current_task[0] * 10    # 还原原始值， 动作映射关系
        deadline = self.current_task[1]

        # 根据动作类型计算时间延迟和能量损耗
        if action == 0:
            T_c = cpu_need / self.local_C                       # 计算时间延迟
            T_t = deadline / self.B_1_2                         # 传输时间延迟,传输策略还需要优化
            E_c = cpu_need * self.d_local
        elif action == 1:
            T_c = cpu_need / self.edge_C
            T_t = cpu_need / self.B_0_1
            E_c = cpu_need * self.d_edge
        elif action == 2:
            T_c = cpu_need / self.cloud_C
            T_t = cpu_need / self.B_0_2
            E_c = cpu_need * self.d_cloud

        match action:
            case 0:
                T_c = cpu_need / self.local_C                       # 计算时间延迟
                T_t = deadline / self.B_1_2                         # 传输时间延迟,传输策略还需要优化
                E_c = cpu_need * self.d_local
            case 1:
                T_c = cpu_need / self.local_C                       # 计算时间延迟
                T_t = deadline / self.B_1_2                         # 传输时间延迟,传输策略还需要优化
                E_c = cpu_need * self.d_local
            case 2:
                T_c = cpu_need / self.local_C                       # 计算时间延迟
                T_t = deadline / self.B_1_2                         # 传输时间延迟,传输策略还需要优化
                E_c = cpu_need * self.d_local
            case _:
                raise ValueError("Invalid action")


        compute_power = self.server_compute_power[action] # 选择服务器能力
        latency = self.server_latency[action]

        # 简化的任务完成时间计算，计算时间
        compute_time = cpu_need / compute_power
        total_delay = compute_time + latency

        # Energy_comsumption = E_c + self.alpha * T_c + self.beta * T_t # 能量消耗
        # delta = 0.1 # 能耗占比
        Energy_comsumption = E_c 
        T_total = T_c + T_t
        Total_cost = self.delta * Energy_comsumption + total_delay # 任务完成时间和能量消耗的加权和

        # reward = - total_delay
        reward = - Total_cost
        done = False                                       # 每次只处理一个任务，回合就结束，是实际就发送一次而已

        info = {"total_delay": total_delay}               #提示信息
        self.current_task = self._generate_task()
        return self.current_task, reward, done, False, info

    def render(self):
        pass  # 可以根据需要添加可视化



# 2. 定义一些论文的函数

# 3. 神经网络模型的定义
import torch
import torch.nn as nn
import torch.nn.functional as F

# cuda 设备的使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
            
        return x
        

class DQN:
    def __init__(self, state_dim=None, action_dim=None, discount=0.9):
        self.discount = discount
        self.Q = Qnet(state_dim, action_dim).to(device)
        self.target_Q = Qnet(state_dim, action_dim).to(device)
        self.target_Q.load_state_dict(self.Q.state_dict()) # copy the parameters from Q to target_Q
        # self.target_Q.eval() # set the target Q network to evaluation mode

    def get_action(self, state):
        qvals = self.Q(state)
        # print("qvals: ", qvals)
        action = qvals.argmax() % qvals.size(1) # index is action, value is q-function value
        return action

    def compute_loss(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        # notice the shape of input and output!!!
        # print("s_batch: ", s_batch.shape)
        # print("a_batch: ", a_batch.shape)
        # print("a_type", type(a_batch[0].item()))
        # print("hello")

        qvals = self.Q(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1) # manipulate the data, and pick what we want
        # next_qvals = self.Q(next_s_batch)
        # next_qvals = self.target_Q(next_s_batch)
        next_qvals = self.target_Q(next_s_batch).max(dim=1)[0]
        loss = F.mse_loss(r_batch + self.discount * next_qvals * (1 - d_batch), qvals)

        return loss
    
    def softupdate_target(self, tau=0.01): # soft update the target Q network
        for target_param, param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        


from dataclasses import dataclass, field 

@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, n):
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=n)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        return state, action, reward, done, next_state

# 4. 内模型，注意一个 episode 形成一个轨迹，作为元学习的一个参数


# 5. 外模型，需要使用 pytorch 的保存神经网络参数功能，否则无法记录训练的效果, episode 可以存到文件中


# 6. 综合任务卸载模型

# 7. 强化学习，环境，动作，状态，奖励函数

# 数据存储，神经网络参数存储



# 论文中的参数
# v_i 的范围没有给我啊，多少个 device，多少个 server
# 如果知道 workflow, 那么相当于已知一个完整的交互过程 episode
# 又要利用经验，还要保证实时性和灵活的，这是一个两难困境

# 1. 导入一些会使用的库，numpy, torch, matplotlib, gymnasium
import gymnasium as gym
# from TaskOffloadEnv import TaskOffloadEnv # 一个是文件名，一个是类名

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn        # 这个是 torch 的神经网络库，torch.nn 里面有很多神经网络可以使用
import torch.optim as optim  # 这个是 torch 的优化器，torch.optim 里面有很多优化器可以使用
import torch.nn.functional as F # 这个是 torch 的函数库，torch.nn.functional 里面有很多函数可以使用

def task_train(env, meta_dqn, meta_loss, meta_optimizer): # 训练一个 task 的函数
    # pass
    meta_optimizer.zero_grad() # 清空梯度

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.98
    epsilon = eps_start

    # env = env_list[0] # 元学习需要改变任务的参数，或者说环境的参数
    replay_buffer = ReplayBuffer(10_000) # mate 一个 buffer, 每个 task 一个 buffer, 还是一个 buffer(封装成一个类会好一点)


    #: 设置不同的 task，整数随机，浮点数随机，等比数列随机
    #: 如果要动态修改环境参数，就需要封装起来，且有接口或者参数修改
    dqn_copy = DQN(2, 3)
    dqn_copy.Q.load_state_dict(meta_dqn.Q.state_dict()) # copy the parameters from meta_dqn to dqn_copy
    # loss = 
    optimizer = optim.Adam(dqn_copy.Q.parameters(), lr=0.001) # 优化器，使用 Adam 优化器，学习率 0.001

    num_episodes = 800 # 总共的 episode 数量， 凑成一个 task 的训练样本，这只是普通的强化学习
    episode_reward_list = []
    episode_loss_list = []
    

    for episode in range(num_episodes):
        episode_reward = 0.0
        episode_loss = 0.0
        obs, info = env.reset() # 元学习和元编程很类似，都是参数模板化
        done = False

        n = 35
        k = 5
        max_step = n + k # 每个 episode 的 step 大小
        count = 0

        # obs = env.reset()

        # print(f"episode{episode}:")
        while count < max_step:
            count += 1
            if (random.random() < epsilon):
                action = env.action_space.sample()  # 用随机策略演示
            else:
                # action = 
                # print("obs: ", obs)
                action = dqn_copy.get_action(torch.tensor([obs]).to(device)) # 这里的 obs 是当前状态， obs_ 是下一个状态
            # x = torch.randn(5, 2)
            # qnet = Qnet(2, 3)
            # y = qnet(x)
            # loss = 
            # if count % 5 == 0 and count > 5:
            if count % 5 == 0 and count > 5:
                s_b, a_b, r_b, d_b, s_b_ = replay_buffer.sample(5) # 采样数据
                s_b = torch.tensor(s_b, dtype=torch.float32).to(device)
                a_b = torch.tensor(a_b, dtype=torch.int64).to(device) # 索引需要整数
                r_b = torch.tensor(r_b, dtype=torch.float32).to(device)
                d_b = torch.tensor(d_b, dtype=torch.float32).to(device)
                s_b_ = torch.tensor(s_b_, dtype=torch.float32).to(device)
                print("trainning...")
                loss = dqn_copy.compute_loss(s_b, a_b, r_b, d_b, s_b_)
                episode_loss += loss.item() # 计算平均损失

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            obs_, reward, done, truncated, info = env.step(action)
            # print(f"obs is : {obs}")
            
            # print(f"Action: {action}, Reward: {reward}, Info: {info}")
            episode_reward += reward

            replay_buffer.push(obs, action, reward, 0, obs_)
            obs = obs_ # 这里的 obs_ 是下一个状态， obs 是当前状态
            
            # with open("episode.txt", "a+") as f:
            #     f.write(f"episode{episode}:") # 没有换行
            #     f.write(f"Action: {action}, Reward: {reward}, Info: {info}\n")

        epsilon = max(eps_end, epsilon * eps_decay) # 衰减 epsilon     
        episode_reward_list.append(episode_reward / max_step) # 计算平均奖励      
        episode_loss_list.append(episode_loss / max_step) # 计算平均损失
        # print("\n")

        # if episode % 20 == 0: # 每 20 个 episode 更新 meta model
        #     # meta_pred = dqn_copy.Q(torch.tensor([obs]).to(device))
        #     pass # update the meta model

    # 持久化数据，或者持久化神经网络参数模型，replay_buffer 在什么时候清空，尤其是元学习会有不同的任务，不同任务的 replay_buffer 是不同的
    # each task 训练就是普通的强化学习（从头，制定的数据，对当前环境参数的最优策略）
    # meta_dqn.
    # 1. reptile method, 先训练一个 task，然后再训练下一个 task，最后更新 meta model
    # for meta_param, param in zip(meta_dqn.Q.parameters(), dqn_copy.Q.parameters()):
    #     meta_param.data.copy_(meta_param.data + 0.1 * (param.data - meta_param.data)) # 这里的 0.1 是学习率，元学习的学习率

    # 取新的 20 episode，太多了 作为 query 数据，但是神经网络的更新是 step 层面啊
    # 1 episodes = 40 transition, 20 episodes = 800 transition, 容易对应过拟合，且计算量大
    query_episodes = 5
    for query_episode in range(query_episodes):
        # query_max_step = 40
        s_b, a_b, r_b, d_b, s_b_ = replay_buffer.sample(40) # 采样数据
        s_b = torch.tensor(s_b, dtype=torch.float32).to(device)
        a_b = torch.tensor(a_b, dtype=torch.int64).to(device) # 索引需要整数
        r_b = torch.tensor(r_b, dtype=torch.float32).to(device)
        d_b = torch.tensor(d_b, dtype=torch.float32).to(device)
        s_b_ = torch.tensor(s_b_, dtype=torch.float32).to(device)
        query_loss = dqn_copy.compute_loss(s_b, a_b, r_b, d_b, s_b_)
        # print("query_loss: ", query_loss)
        meta_loss += query_loss

    meta_loss = meta_loss / query_episodes



    state, action, reward, done, state_ = replay_buffer.sample(5)
    # print(state)
    # print(action)
    # print(reward)
    # print(done)
    # print(state_)

    # test for painting

    # print(type(state[0]))
    # print(type(action[0])) #  list of ndarray type, could be used for painting
    # print(type(reward))
    # print(type(done))
    # print(type(state_[0]))

    x = np.linspace(0, num_episodes, num_episodes) # 画图的 x 轴
    # y = np.linspace(0, 1, num_episodes) # 画图的 y 轴
    # print(y)

    with open(f"./DMRO_data/delta_{env.delta}_dmro.txt", "w") as f:
        for i in range(num_episodes):
            f.write(f"{episode_reward_list[i]}\n")

    plt.figure(figsize=(5, 5))
    plt.plot(x, episode_reward_list, label='test for training')
    # plt.plot(x, episode_loss_list, label='loss')
    # 保存图像，训练非阻塞

    plt.legend()
    plt.title('Reward')

    plt.grid(True)
    plt.show()

    return meta_loss


def make_env_array(): # 制作 环境数组
    env_array = []
    for i in range(1, 3):
        env = TaskOffloadEnv(num_servers = 3) # TODO:
        env_array.append(env)
    return env_array

# env = TaskOffloadEnv(num_servers=3) # 元学习需要改变任务的参数值，但是如果数量改变导致神经网络结构变化就不能够复用了
def sample_env(env_list):
    env = random.choice(env_list)
    return env

if __name__ == "__main__":

    env_list = make_env_array()
    # print(len(env_list))
    # assert len(env_list) == 3, "env list length is not 2"


    meta_dqn = DQN(2, 3) # 元学习的神经网络，注意和 task 中的区分开
    meta_loss = 0.0
    meta_loss_list = []
    meta_optimizer = optim.Adam(meta_dqn.Q.parameters(), lr=0.001) # 元优化器，使用 Adam 优化器，学习率 0.001


    num_task_episodes = 1
    for i in range(num_task_episodes):

        env = sample_env(env_list)
        task_loss = task_train(env, meta_dqn, meta_loss, meta_optimizer) # 将元神经网络也传入, 浅拷贝，传的是引用
        meta_loss += task_loss

    # 状态维度变化，动作维度变化，任务结构变化，元学习是不太好的
    # TODO: 所有 task 完成后，更新 meta model, 并且保存模型参数

    meta_loss = meta_loss / num_task_episodes # 计算平均损失


    meta_optimizer.zero_grad() # 清空梯度
    meta_loss.backward() # 反向传播
    meta_optimizer.step() # 更新参数

    torch.save(meta_dqn.Q.state_dict(), "meta_model.pth") # 保存模型参数

    # new_model = DQN(2, 3)
    # new_model.Q.load_state_dict(torch.load("meta_model.pth"))

    # new_model.Q.eval() # 设置为评估模式, 不进行训练