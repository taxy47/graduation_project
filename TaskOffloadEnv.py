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

class user(): # 用户
    def __init__(self):
        self.v_i = random.uniform(1.0, 10.0) # 任务大小
        self.e_i = random.uniform(0.1, 1.0)

class TaskOffloadEnv(gym.Env):
    def __init__(self, num_servers=3):
        super(TaskOffloadEnv, self).__init__()
        self.num_servers = num_servers

        # 任务状态空间：每个任务有 CPU 需求 (0-10), deadline (0-1), 那么是否还要进行 action mapping
        self.observation_space = spaces.Box(low=2.0, high=5.0, shape=(2,), dtype=np.float32)

        # 动作空间：选择将任务分配到哪个边缘服务器, 0: 本地计算, 1: 边缘服务器, 2: 云服务器
        self.action_space = spaces.Discrete(num_servers)

        # 边缘服务器特性（计算能力和网络延迟), 其实需要计算得知而不是随机生成
        self.server_compute_power = np.random.uniform(5, 10, size=(num_servers,))
        self.server_latency = np.random.uniform(0.05, 0.2, size=(num_servers,))

        self.cloud_C = 1000 # compute capability
        self.edge_C = 100 # compute capability
        self.local_C = 1 # compute capability

        # Todo: 放在实现类中，并且使用字典或者列表来存储
        self.B_0_1 = 800 # 800 MBps， 本地和边缘服务器之间的带宽
        self.B_1_2 = 200 # 200 MBps， 边缘服务器和云服务器之间的带宽
        self.B_0_2 = 10 # 10 MBps， 云服务器传输很慢，计算很强

        self.d_cloud = 0.1 # J/MB
        self.d_edge = 0.15 # J/MB
        self.d_local = 0.3 # J/MB

        self.alpha = 0.5
        self.beta = 0.5

    def reset(self, seed=None, options=None): # 外部会初始时调用
        super().reset(seed=seed)
        self.current_task = self._generate_task()
        return self.current_task, {}

    def _generate_task(self):  # 生成任务
        cpu = np.random.uniform(1, 10)
        deadline = np.random.uniform(0.1, 1.0)
        return np.array([cpu / 10.0, deadline], dtype=np.float32)

    def step(self, action):
        cpu_need = self.current_task[0] * 10    # 还原原始值， 动作映射关系
        deadline = self.current_task[1]

        # 根据动作类型计算时间延迟和能量损耗,可以直接返回，也可以继续添加和复杂化
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

        reward = 1.0 if total_delay <= deadline else -1.0
        done = True                                       # 每次只处理一个任务，回合就结束

        info = {"total_delay": total_delay}               #提示信息
        self.current_task = self._generate_task()
        return self.current_task, reward, done, False, info

    def render(self):
        pass  # 可以根据需要添加可视化



# 论文中的参数
# v_i 的范围没有给我啊，多少个 device，多少个 server
# 如果知道 workflow, 那么相当于已知一个完整的交互过程 episode
# 又要利用经验，还要保证实时性和灵活的，这是一个两难困境

if __name__ == "__main__":
    env = TaskOffloadEnv()
    obs, info = env.reset()
    print("obs: ", obs)
    print("info: ", info)