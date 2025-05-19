import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
#算法


class SAC_AVERAGE(object):
    # num_imputs、action_space分别为状态维度和状态动作
    def __init__(self, num_inputs, action_space, args):

        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval  # the interval of update
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # 可学习的平均奖励率 rho
        # 通常初始化为0或一个较小的值。可以为 rho 设置独立的学习率，例如 args.lr_rho
        self.rho = torch.zeros(1, requires_grad=True, device=self.device)
        # 建议为 rho 使用一个较小的学习率, 例如 args.lr * 0.1 或定义一个新的 args.lr_rho
        self.rho_optim = Adam([self.rho], lr=args.lr)


        # 这个Q网络是状态-动作价值函数
        # Q网络初始化，将q1,q2参数、网络可训练参数、网络结构传递给critic Q
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)   # state num; action num; hidden size num.
        # 利用adam算法对Q网络进行更新收敛，学习率为lr
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        # 构造critic Q网络相同的target Q网络
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)    # copy directly

        # 高斯策略
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            # 判断是否对参数正则化调整
            if self.automatic_entropy_tuning is True:
                # 计算目标熵，定义为-动作空间维度
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                # 初始化对数熵系数并设置为可训练参数
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            # 高斯策略初始化
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            
        #TODO 添加一个自然策略梯度算法

        else: # Deterministic Policy
            self.alpha = 0 # 确定性策略没有熵项
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state) # 对于确定性策略, sample 可能返回均值
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory，从memory中随机batch_size个样本
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        # 数据转换为Pytorch张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)  # mask 为1表示非终止状态, 为0表示终止状态

        # --- Critic 更新 ---
        with torch.no_grad(): # 目标值的计算不应影响梯度
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target_vals, qf2_next_target_vals = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target_vals = torch.min(qf1_next_target_vals, qf2_next_target_vals)
            # 下一状态的软V值: V(s') = Q_target(s',a') - alpha * log_pi(a'|s')
            next_v_target = min_qf_next_target_vals - self.alpha * next_state_log_pi # self.alpha 在此上下文中应视为常数或来自log_alpha.exp().detach()
            # 差分Q值的目标: r - rho + V(s') (如果不是终止状态)
            # mask_batch 为1表示非终止状态, 为0表示终止状态。如果 mask_batch 为0, next_v_target 部分被置零。
            next_q_value_target = reward_batch - self.rho.detach() + mask_batch * next_v_target

        qf1, qf2 = self.critic(state_batch, action_batch)  # 当前Q值
        qf1_loss = F.mse_loss(qf1, next_q_value_target)
        qf2_loss = F.mse_loss(qf2, next_q_value_target)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # --- Policy 更新 ---
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi) # 当前策略动作的Q值
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # 策略目标: 最大化 E[Q(s,a) - alpha * log_pi(a|s)]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Alpha (熵温度) 更新 ---
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha).to(self.device) # 使用 self.device

        # --- Rho (平均奖励率) 更新 ---
        # Rho 的目标是 E[r + V_target(s') - V_current(s)]
        # V_target(s') = next_v_target (为 critic 更新计算的，已分离梯度)
        # V_current(s) = (min_qf_pi - self.alpha * log_pi).detach() (来自当前策略和critic)
        with torch.no_grad(): # rho 的目标计算不应有梯度回传
            # 当前状态的V值 (需要分离梯度)
            v_s_current_detached = (min_qf_pi.detach() - self.alpha.detach() * log_pi.detach())
            # next_v_target 是在 critic 的 no_grad块中计算的，所以它已经是分离的
            rho_update_target = (reward_batch + mask_batch * next_v_target - v_s_current_detached).mean()

        # Rho 损失: (rho - target_rho)^2
        # self.rho 是一个标量, rho_update_target 也是一个标量均值
        rho_loss = F.mse_loss(self.rho, rho_update_target.expand_as(self.rho))

        self.rho_optim.zero_grad()
        rho_loss.backward()
        self.rho_optim.step()


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), self.rho.item(), rho_loss.item()
    
    # Save model parameters 保存网络模型参数
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path, map_location='cpu'))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location='cpu'))
