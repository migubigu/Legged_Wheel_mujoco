import numpy as np
import random
import math
import gym
import torch
import threading
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import argparse
from sac import SAC
from replay_memory import ReplayMemory
from pad import control_gamepad
import envs.register
from utils import reset_pad



#训练好的神经网络模型记录在models文件夹中，通过此调用测试

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Biped-v0",
                    help='Mujoco Gym environment (default: Biped-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
parser.add_argument('--control_if', type=bool, default=True,
                    help='control if use gamepad (default: True)')
args = parser.parse_args()

# Environment, Agent
env = gym.make(args.env_name)
env.cfg(control_if=args.control_if, cuda_if=args.cuda)
agent = SAC(env.observation_space.shape[0], env.action_space, args)

agent.load_model('models/sac_actor_Biped-v0_', 'models/sac_critic_Biped-v0_')

avg = 0.0
res = []


# # 线程控制
# reset_flag = None
# stop_thread = False
# def get_commands_thread(pad):
#     global commands, reset_flag, stop_thread
#     while not stop_thread:
#         commands, reset_flag = pad.get_commands()

# 初始化遥控器
command_cfg, command_scale = reset_pad()
pad = control_gamepad(command_cfg, command_scale)

# command_thread = threading.Thread(target=get_commands_thread, args=(pad,))
# command_thread.start()


# Testing loop
def testSAC():
    for i in range(100):
        commands, reset_flag = pad.get_commands()
        env.control_input(commands_in = commands)       # 将pygame指令传递给env
        state = env.reset()
        ret = 0.0
        for t in count():
            commands, reset_flag = pad.get_commands()
            env.control_input(commands_in=commands)
            if reset_flag:
                state = env.reset()
                print("reset")
                break
            env.unwrapped._resample_commands()
            # 更新渲染画面
            env.render()
            action = agent.select_action(state, evaluate=True)

            nextState, reward, done, _ = env.step(action)
            ret += reward
            state = nextState
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                res.append(ret)
                break
        print(ret)
    avg = np.average(res)
    
    return res, avg


res, avg = testSAC()


