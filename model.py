import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
#神经网络模型

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights 初始化权重
def weights_init_(m):
    if isinstance(m, nn.Linear):                            # 判断m是否是nn.Linear类型
        torch.nn.init.xavier_uniform_(m.weight, gain=1)     #xavier_uniform_根据输入m对权重进行初始化
        torch.nn.init.constant_(m.bias, 0)                  #constant_将偏置参数初始化为0

# 但是项目中似乎并没有使用
# 状态价值
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        # 初始化定义线性回归格式
        self.value = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        #apply()调用weights_init_函数，对上面的三个线性层遍历初始化
        self.apply(weights_init_)

    def forward(self, state):
        # F.relu()数学形式为f(x)=max(0,x)，对函数进行激活
        x = self.value(state)
        return x

# nn.Module为继承类
# 状态-动作价值
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        # 调用父类nn.Module的__init__方法初始化
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        # cat()将state和action合并，列增加
        xu = torch.cat([state, action], 1)
        #relu函数激活神经网络
        x1 = self.q1(xu)
        x2 = self.q2(xu)

        return x1, x2


# 高斯策略
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.gaussian = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 为什么均值和标准差也要用神经网络？
        # 定义动作均值
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # 定义动作对数标准差
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling，若没有action_space，则定义初始动作空间;否则定义动作空间
        if action_space is None:
            # torch.tensor()将数据转换为张量，定义了初始动作空间
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            # torch.FloatTensor()将数据转换为张量
            # 统计动作变量，用于标准化动作空间
            # action_scale返回动作空间的幅度，action——bias返回动作空间的偏置（均值）
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
    def forward(self, state):
        x = self.gaussian(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # 将log_std限制在LOG_SIG_MIN和LOG_SIG_MAX之间
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    # 从策略网络得到的mean和log_std高斯分布，返回实际空间动作-概率和动作分布均值
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # 构造均值mean，标准差为std的正态分布
        normal = Normal(mean, std)
        # 利用normal概率分布生成随机数样本
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)   # between -1 1
        # 通过y_t将标准化的动作空间映射到原始动作空间
        action = y_t * self.action_scale + self.action_bias  # from env
        # 计算随机样本在normal概率分布中x_t对应概率的对数
        log_prob = normal.log_prob(x_t)   #
        # Enforcing Action Bound
        # 由于tanh的非线性变换将改变采样x_t的概率值，因此要对其进行修正
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # 对各维度概率求和得到整个动作的联合对数概率
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias   # the mean action value
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

# 确定性策略
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.deterministic = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.deterministic(state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
