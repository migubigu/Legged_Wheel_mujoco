import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R
import os
import random
import torch
# 封装环境重置、步态环境更新状态反馈等。主要功能是通过python接口封装mujoco仿真器中机器人的信息

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# 定义一个仿真环境
class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        # 读取环境路径(这是一个完整的环境，它包括了机器人、世界、接触、碰撞、关节、骨骼·······)
        # ctrl_cost_weight是控制成本的权重
        xml_file=os.path.join(os.path.join(os.path.dirname(__file__),
                                'asset', "Legged_wheel3.xml")),
        healthy_reward=1.0,
        healthy_z_range=0.05,
        reset_noise_scale=0.1,
    ):
        # 将__init__方法传递给父类,EzPickle是用来将自定义环境中的参数数列化的方法
        utils.EzPickle.__init__(**locals())

        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.action_dim = 6
        self.history_actions_for_obs = [np.zeros(self.action_dim) for _ in range(2)]

        self.cfg()
        # 这里的5是什么意思，意思是每调用一次step()，仿真会进行五个物理仿真步
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property # 创建只读属性，防止属性被修改
    # 计算健康奖励
    def healthy_reward(self):
        return (
            float(self.is_healthy) * self._healthy_reward
        )
    
    
    # 目标指令初始化
    def cfg(self, control_if=False, cuda_if=False):
        self.control_if = control_if
        self.device = torch.device("cuda" if cuda_if else "cpu")
        self.commands = torch.zeros(2).to(self.device)             # 控制指令，线速度，旋转角速度
        # 最大指令范围
        self.lin_vel_x = [-1.2, 1.2]
        self.ang_vel_z = [-0.7, 0.7]
        # 初始课程阶段的指令范围
        self.current_lin_vel_x_range = [-0.3, 0.3]
        self.current_ang_vel_z_range = [-0.2, 0.2]
        self.min_stand = 0.1
        self.target_height = 0.15
        self.shin_height_min = 0.06
        self.ang_range = np.pi / 6
        # 课程学习参数
        self.curriculum_stage = 0
        self.max_curriculum_stage = 3 # 例如，0, 1, 2, 3 四个阶段
        self.stage_reward_thresholds = [3.5, 4.0, 4.5] # 提升到下一阶段所需的平均奖励阈值 (需要根据奖励函数实际调整)
        self.stage_episodes_stable = 10 # 需要连续多少个episode满足阈值才能升级
        self.stable_episode_count = 0
        self.command_step = 500
        self.max_r_vel = 0.4

    # 获取指令
    def control_input(self,commands_in=None):
        if self.control_if:
            self.commands_input = commands_in
        else:
            self.commands_input = None

    # 生成控制指令
    def _resample_commands(self):
        if self.control_if:
            self.commands[0] = self.commands_input[0]
            self.commands[1] = self.commands_input[1]
        else:
            current_lin_vel_range = self.current_lin_vel_x_range
            current_ang_vel_range = self.current_ang_vel_z_range
            if self.sim.data.time == 0:
                self.target_lin_vel = random.uniform(*current_lin_vel_range)
                self.target_ang_vel = random.uniform(*current_ang_vel_range)
            else:
                if self.curriculum_stage >= 2:                                          # 在高课程下，可能会维持当前状态
                    if random.random() < 0.2:
                        self.target_lin_vel = random.uniform(*current_lin_vel_range)
                        self.target_ang_vel = random.uniform(*current_ang_vel_range)
                    else: 
                        self.target_lin_vel = np.clip(
                            self.target_lin_vel + random.uniform(-0.05, 0.05), 
                            *current_lin_vel_range)
                        self.target_ang_vel = np.clip(
                            self.target_ang_vel + random.uniform(-0.02, 0.02), 
                            *current_ang_vel_range)
                else: 
                    self.target_lin_vel = np.clip(
                        self.target_lin_vel + random.uniform(-0.1, 0.1),
                        *current_lin_vel_range)
                    self.target_ang_vel = np.clip(
                        self.target_ang_vel + random.uniform(-0.05, 0.05),
                        *current_ang_vel_range)

            alpha = 0.1
            if self.curriculum_stage >=2 and (abs(self.target_lin_vel) > 0.6 * current_lin_vel_range[1] or abs(self.target_ang_vel) > 0.6 * current_ang_vel_range[1]):
                alpha = 0.05
            self.commands[0] = (1-alpha)*self.commands[0] + alpha*self.target_lin_vel
            self.commands[1] = (1-alpha)*self.commands[1] + alpha*self.target_ang_vel
        self.commands = self.commands.to(self.device)
    
    @property  # 是否倾倒（通过质心到达最低健康高度、碰到大腿以上作为是否倾倒依据）、get_body_com是MujocoEnv中的一种方法，获取“base_link”的质心位置
    def is_healthy(self):
        min_z = self._healthy_z_range
        # 计算相对高度
        ang_yx = self.get_agent_euler()[1:]                  # 对xy轴位姿限制
        # is_healthy = ((self.get_body_com("base_link")[2]-self.get_body_com("left_wheel")[2]) > min_z) and (not self.bump_base()) and (abs(ang_yx[0]) < self.ang_range) and (min(abs(ang_yx[1]),abs(ang_yx[1] - np.pi),abs(ang_yx[1] + np.pi)) < self.ang_range) and (shin_height > self.shin_height_min)  # 判断是否倾倒
        is_healthy = ((self.get_body_com("base_link")[2]-self.get_body_com("left_wheel")[2]) > min_z) and (not self.bump_base()) and (abs(ang_yx[0]) < self.ang_range) and (min(abs(ang_yx[1]),abs(ang_yx[1] - np.pi),abs(ang_yx[1] + np.pi)) < self.ang_range)
        return is_healthy

    @property #判断是否结束
    def done(self):
        done = not self.is_healthy
        return done

    # 碰到大腿以上
    # sim.data.ncon表示当前仿真时间步中检测到的接触对的数量
    # sim.data.contact[i]表示第i个接触对的信息
    # sim.model.geom_id2name(contact.geom1)表示第i个接触对中的第一个几何体的名称
    # 判断goem1和goem2是否为大腿以上的几何体，如果是则返回True，否则返回False
    def bump_base(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if (geom1 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                          'right_thigh1', 'right_thigh2', 'right_thigh3']) or (
                    geom2 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                              'right_thigh1', 'right_thigh2', 'right_thigh3']):
                return True
        return False

    # 输出接触地面
    def wheel_contact(self):
        left_contact = False
        right_contact = False
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(c.geom1) == "left_wheel" or self.sim.model.geom_id2name(c.geom2) == "left_wheel":
                left_contact = True
            if self.sim.model.geom_id2name(c.geom1) == "right_wheel" or self.sim.model.geom_id2name(c.geom2) == "right_wheel":
                right_contact = True
            if left_contact and right_contact:
                break
        if left_contact and right_contact:
            return 2
        elif left_contact or right_contact:
            return 1
        else:
            return 0

    # 获取当前Agent的位姿
    def get_agent_euler(self):
        w, x, y, z = torch.from_numpy(self.quat_imu).to(self.device)  # 获取四元数
        # 计算欧拉角 (yaw, pitch, roll)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        pitch = torch.arcsin(2.0 * (w * y - z * x))
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))

        return [yaw, pitch, roll]  # 返回为弧度制
    # 四元数计算
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.tensor([w, x, y, z])
    
    # 课程学习
    def class_learning(self, average_episode_reward, current_episode_number):
        if self.curriculum_stage < self.max_curriculum_stage:
            # 检查是否达到提升到下一阶段的平均奖励阈值
            if average_episode_reward > self.stage_reward_thresholds[self.curriculum_stage]:
                self.stable_episode_count += 1
            else:
                self.stable_episode_count = 0

            if self.stable_episode_count >= self.stage_episodes_stable:     # 如果连续多个episode表现稳定且达标，则提升阶段
                self.curriculum_stage += 1
                self.stable_episode_count = 0
                print(f"Episode {current_episode_number}: Curriculum ADVANCED to Stage {self.curriculum_stage} (Avg Reward: {average_episode_reward:.2f})")
                self._update_curriculum_parameters()
        else:
            if current_episode_number % 100 == 0: # 每100幕打印一次，表示已在最高级
                 print(f"Episode {current_episode_number}: Curriculum at MAX Stage {self.curriculum_stage} (Avg Reward: {average_episode_reward:.2f})")
        return self.curriculum_stage

    # 课程控制指令更新
    def _update_curriculum_parameters(self):
        if self.control_if:
            self.current_lin_vel_x_range = self.lin_vel_x # 使用最大范围
            self.current_ang_vel_z_range = self.ang_vel_z # 使用最大范围
            self.command_step = 200
            self.max_r_vel = 0.1
        else:
            if self.curriculum_stage == 0:
                self.current_lin_vel_x_range = [-0.3, 0.3]
                self.current_ang_vel_z_range = [-0.2, 0.2]
                self.command_step = 500
                self.max_r_vel = 0.4
            elif self.curriculum_stage == 1:
                self.current_lin_vel_x_range = [-0.6, 0.6]
                self.current_ang_vel_z_range = [-0.4, 0.4]
                self.command_step = 400
                self.max_r_vel = 0.3
            elif self.curriculum_stage == 2:
                self.current_lin_vel_x_range = [-0.9, 0.9]
                self.current_ang_vel_z_range = [-0.6, 0.6]
                self.command_step = 200
                self.max_r_vel = 0.2
            elif self.curriculum_stage >= self.max_curriculum_stage: # 最高阶段
                self.current_lin_vel_x_range = self.lin_vel_x # 使用最大范围
                self.current_ang_vel_z_range = self.ang_vel_z # 使用最大范围
                self.command_step = 200
                self.max_r_vel = 0.2

    # 执行仿真中的一步
    def step(self, action):
        self.do_simulation(action, self.frame_skip)                             # 更新仿真
        self.action = action
        if (int((self.sim.data.time/0.002)+0.5)) % (self.command_step) == 0:                   # 更新控制指令
            self._resample_commands()
        #获取先前动作
        self.history_actions_for_obs[1] = self.history_actions_for_obs[0].copy()
        self.history_actions_for_obs[0] = self.action.copy()

        observation = self._get_obs()                                           # 获取观察值
        self.lin_vel_agent_xyz = torch.tensor(self.vel_agent, dtype=torch.float32).to(self.device)                  # 获取Agent的线速度
        self.lin_vel_agent_local = self.lin_vel_agent_xyz[0]

        # 计算奖励

        rewards_info = {}
        r_vel_weight = 1.0
        if self.curriculum_stage >= 2:              # 在高阶课程中，更看重速度跟踪
            r_vel_weight = 1.3
        if self.curriculum_stage == self.max_curriculum_stage:
            r_vel_weight = 1.6

        rewards_info['r_vel'] = self._reward_tracking_vel().cpu()* r_vel_weight
        rewards_info['r_vel_z'] = self._reward_lin_vel_z().cpu()*0.2
        rewards_info['r_ang_xy'] = self._reward_ang_vel_xy().cpu()*0.2
        rewards_info['r_torque'] = self._reward_torque().cpu()*0.3
        rewards_info['r_gravity'] = self._reward_gravity().cpu()*0.5
        rewards_info['r_similar_legged'] = self._reward_similar_legged().cpu()*0.4
        rewards_info['r_stand'] = self._reward_nominal_state().cpu()*0.05
        rewards_info['r_height'] = self._reward_base_height().cpu()*0.1
        rewards_info['r_energy'] = self._reward_energy().cpu()*0.2
        rewards_info['r_elur'] = self._reward_base_ang_xy().cpu()*0.5
        # rewards_info['r_smooth'] = self._reward_action_smoothness().cpu()*0.1
        rewards_info['r_wheel_contect'] = self._reward_wheel_contact().cpu()*0.2
        rewards_info['r_ang_vel_enhance'] = self._reward_ang_vel_enhance().cpu()*r_vel_weight
        rewards_info['r_lin_vel_enhance'] = self._reward_lin_vel_enhance().cpu()*r_vel_weight
        rewards_info['r_healthy'] = self.healthy_reward*0.8
        
        reward = sum(rewards_info.values()) # 计算总奖励
        done = self.done                        # 只要没死亡就可以一直仿真

        info = {"rewards": rewards_info} # 记录奖励信息
        return observation.cpu().numpy(), reward.tolist(), done, info
    

    # 获取当前状态的观察值
    # sim.data.qpos.flat.copy()表示当前所有关节位置
    # sim.data.qvel.flat.copy()表示当前所有关节速度
    def _get_obs(self):
        self.position = self.sim.data.qpos.flat.copy() # 身体部位的位置
        self.velocity = self.sim.data.qvel.flat.copy() # 所有关节的线速度
        self.torques = self.sim.data.actuator_force.flat.copy() # 所有关节的扭矩
        # imu传感器获取的数据似乎都是局部坐标系下的
        gyro_id = self.model.sensor_name2id("Body_Gyro")
        vel_id = self.model.sensor_name2id("Body_Vel")
        # 获取Agent上传感器imu的速度、角速度和位姿
        self.vel_agent = self.sim.data.sensordata[self.sim.model.sensor_adr[vel_id]:self.sim.model.sensor_adr[vel_id] + self.sim.model.sensor_dim[vel_id]]
        self.gyro_agent= self.sim.data.sensordata[self.sim.model.sensor_adr[gyro_id]:self.sim.model.sensor_adr[gyro_id] + self.sim.model.sensor_dim[gyro_id]]
        self.quat_imu = self.position[3:7]
        # self.acc_agent = self.sim.data.sensordata[self.sim.model.sensor_adr[acc_id]:self.sim.model.sensor_adr[acc_id] + self.sim.model.sensor_dim[acc_id]]
        # 获取重力投影
        self.gravity = torch.tensor([0, 0, 0, -9.81], dtype=torch.float32).to(self.device)
        quat_conj = torch.tensor([self.quat_imu[0], -self.quat_imu[1], -self.quat_imu[2], -self.quat_imu[3]], dtype=torch.float32).to(self.device)
        on = self.quaternion_multiply(self.quat_imu, self.gravity)
        self.gravity_local = self.quaternion_multiply(on, quat_conj).to(self.device)
        contact_forces = np.array([self.wheel_contact()])
        observations = torch.cat((
            torch.tensor(self.position[3:]).to(self.device),
            torch.tensor(self.velocity).to(self.device),
            torch.tensor(self.torques).to(self.device),
            torch.tensor(self.vel_agent).to(self.device),
            torch.tensor(self.gyro_agent).to(self.device),
            self.gravity_local,
            self.commands,
            torch.tensor(self.history_actions_for_obs[0]).to(self.device),
            torch.tensor(self.history_actions_for_obs[1]).to(self.device),
            torch.tensor(contact_forces).to(self.device)
        ))
        return observations

    # 重置模型
    def reset_model(self):
        if hasattr(self, 'last_action'):
            delattr(self, 'last_action')
        self._resample_commands()
        observation = self._get_obs()
        return observation.cpu().numpy()

    # 可视化查看器
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

# reward函数
    def _reward_tracking_vel(self):
        stand_command = (torch.norm(self.commands) <= self.min_stand)
        lin_vel_error_square = torch.square(self.lin_vel_agent_local - self.commands[0])
        ang_vel_error_square = torch.square(self.gyro_agent[2] - self.commands[1])
        lin_vel_error_abs = abs(self.lin_vel_agent_local - self.commands[0])
        ang_vel_error_abs = abs(self.gyro_agent[2] - self.commands[1])

        r_square = torch.exp(-lin_vel_error_square) + torch.exp(-ang_vel_error_square)
        r_abs = torch.exp(-lin_vel_error_abs * 1.2) + torch.exp(-ang_vel_error_abs * 1.2)
        r = torch.where(stand_command, r_square, r_abs)
        return r
    # 惩罚z轴线速度
    def _reward_lin_vel_z(self):
        r = torch.square(self.lin_vel_agent_xyz[2])
        return torch.exp(-r)
    # 惩罚x，y轴转动
    def _reward_ang_vel_xy(self):
        r=torch.sum(torch.square(torch.from_numpy(self.gyro_agent)[:2]))
        return torch.exp(-r)
    # 惩罚过大力矩
    def _reward_torque(self):
        r=torch.sum(torch.square(torch.from_numpy(self.torques[:2]))) + torch.sum(torch.square(torch.from_numpy(self.torques[3:5])))
        return torch.exp(-r*0.001)
    # 低速奖励维持默认动作
    def _reward_nominal_state(self):
        stand_command = (torch.norm(self.commands) <= self.min_stand)
        r = torch.exp(- torch.sum(torch.abs(torch.tensor(self.position[7:9]))) - torch.sum(torch.abs(torch.tensor(self.position[10:12]))))
        r = torch.where(stand_command, r.to(self.device), torch.zeros_like(r).to(self.device))
        return r

    def _reward_gravity(self):
        # 计算重力对Agent的影响
        r = torch.abs(self.gravity_local[3] - self.gravity[3])
        return torch.exp(-r)
    # 奖励双腿动作相同
    def _reward_similar_legged(self):
        left_leg = torch.tensor(self.position[7:9])
        right_leg = torch.tensor(self.position[10:12])
        r = torch.sum(torch.square(left_leg - right_leg))
        return torch.exp(-r)
    # 奖励保持目标高度
    def _reward_base_height(self):
        wheel_height = torch.tensor([self.get_body_com("left_wheel")[2], self.get_body_com("right_wheel")[2]])
        average_wheel = torch.mean(wheel_height)
        r = torch.square((torch.tensor(self.get_body_com("base_link")[2])-average_wheel) - torch.tensor(self.target_height))
        return torch.exp(-r*5)
    # 惩罚过大动作幅度
    def _reward_energy(self):
        r = torch.sum(torch.abs(torch.tensor(self.action)))
        return torch.exp(-r*0.1)

    def _reward_base_ang_xy(self):
        pitch, roll =self.get_agent_euler()[1:]
        r = torch.square(pitch) + torch.square(roll)
        return torch.exp(-r)
    # 奖励动作连续
    # def _reward_action_smoothness(self):
    #     current_action_tensor = torch.from_numpy(self.action.astype(np.float32)).to(self.device)
    #     if hasattr(self, 'last_action'):
    #         action_diff = torch.mean(torch.abs(current_action_tensor - self.last_action))
    #         self.last_action = current_action_tensor.clone()
    #         return torch.exp(-5*action_diff)
    #     else:
    #         self.last_action = current_action_tensor.clone()
    #         return torch.tensor(1.0)
        
    def _reward_wheel_contact(self):
        contact_status = self.wheel_contact()
        r = torch.tensor(2 - contact_status)
        return torch.exp(-0.5*r)
    
    def _reward_ang_vel_enhance(self):
        ang_vel_error_abs = abs(self.gyro_agent[2] - self.commands[1])
        if ang_vel_error_abs > self.max_r_vel:
            return -0.3*ang_vel_error_abs
        else:
            return torch.exp(-2.0*ang_vel_error_abs)
    
    def _reward_lin_vel_enhance(self):
        lin_vel_error_abs = abs(self.lin_vel_agent_local - self.commands[0])
        if lin_vel_error_abs > self.max_r_vel:
            return -0.3*lin_vel_error_abs
        else:
            return torch.exp(-2.0*lin_vel_error_abs)