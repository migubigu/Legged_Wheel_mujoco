from gym.envs.registration import register
import numpy as np
# 注册Biped-v0，将BipedEnv注册到gym环境中
# entry_point是python模块路径
register(
    id="Biped-v0", # 环境id
    entry_point="envs.biped:BipedEnv", # 环境类入口
    max_episode_steps=100000, # 一个episode的最大步数
    reward_threshold=6000000.0, # 完成任务的奖励阈值
)


