import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TaskOffloadEnv(gym.Env):
    def __init__(self, num_servers=3):
        super(TaskOffloadEnv, self).__init__()
        self.num_servers = num_servers

        # 任务状态空间：每个任务有 CPU 需求 (0-10), deadline (0-1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # 动作空间：选择将任务分配到哪个边缘服务器
        self.action_space = spaces.Discrete(num_servers)

        # 边缘服务器特性（计算能力和网络延迟）
        self.server_compute_power = np.random.uniform(5, 10, size=(num_servers,))
        self.server_latency = np.random.uniform(0.05, 0.2, size=(num_servers,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_task = self._generate_task()
        return self.current_task, {}

    def _generate_task(self):
        cpu = np.random.uniform(1, 10)
        deadline = np.random.uniform(0.1, 1.0)
        return np.array([cpu / 10.0, deadline], dtype=np.float32)

    def step(self, action):
        cpu_need = self.current_task[0] * 10  # 还原原始值
        deadline = self.current_task[1]

        compute_power = self.server_compute_power[action]
        latency = self.server_latency[action]

        # 简化的任务完成时间计算
        compute_time = cpu_need / compute_power
        total_delay = compute_time + latency

        reward = 1.0 if total_delay <= deadline else -1.0
        done = True  # 每次只处理一个任务，回合就结束

        info = {"total_delay": total_delay}
        self.current_task = self._generate_task()
        return self.current_task, reward, done, False, info

    def render(self):
        pass  # 可以根据需要添加可视化
