import random
import numpy as np
# 经验池的定义，capacity为经验池的大小;seed为随机种子,在main.py中初始化
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    # 添加写入状态、动作、奖励、下一个状态和是否结束的标志
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    # 在经验池中随机采集batch_size大小样本
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 用zip(*)先将batch解压，再用map(np.stack)将state等堆叠为numpy数组
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
