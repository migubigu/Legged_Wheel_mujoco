import math
import torch
import matplotlib.pyplot as plt
import numpy as np
#定义函数

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def reset_pad():
    command_cfg = {
        'num_commands': 2,
        'lin_vel_x_range': [-1, 1],
        'ang_vel_range': [-0.5, 0.5]
    }
    command_scale = [1.0, 1.0, 1.0]
    return command_cfg, command_scale

def rewards_output_pic(overall_reward_components_average):
    plt.clf()
    subplot_idx = 1 # 初始化子图索引
    num_rows = 4
    num_cols = 4
    for comp_name in overall_reward_components_average.keys():
        y = overall_reward_components_average[comp_name]
        x = np.arange(len(y))
        plt.subplot(num_rows, num_cols, subplot_idx)
        plt.title(comp_name)
        plt.plot(x, y)
        subplot_idx += 1
    plt.pause(10)
    plt.ioff()  # 关闭画图的窗口