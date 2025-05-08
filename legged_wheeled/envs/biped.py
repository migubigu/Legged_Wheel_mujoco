import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R
import os
import random
import torch
import torchvision
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

        self.cfg()

        
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
        self.cuda_if = cuda_if
        self.device = torch.device("cuda" if self.cuda_if else "cpu")
        self.commands = torch.zeros(2).to(self.device)             # 控制指令，线速度，旋转角速度
        self.lin_vel_x = [-1.2, 1.2]
        self.ang_vel_z = [-1.0, 1.0]
        self.min_stand = 0.1
        self.target_height = 0.125
        self.shin_height_min = 0.06
        self.ang_range = np.pi / 6

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
            if self.sim.data.time == 0:
                self.commands[0] = random.uniform(self.lin_vel_x[0], self.lin_vel_x[1])  # 随机生成x线速度
                self.commands[1] = random.uniform(self.ang_vel_z[0], self.ang_vel_z[1])  # 随机生成z角速度
            else:
                self.commands[0] += random.uniform(-0.1, 0.1)
                self.commands[1] += random.uniform(-0.05, 0.05)
                self.commands[0] = np.clip(self.commands[0], self.lin_vel_x[0], self.lin_vel_x[1])
                self.commands[1] = np.clip(self.commands[1], self.ang_vel_z[0], self.ang_vel_z[1])
        self.commands = self.commands.to(self.device)  


    @property  # 是否倾倒（通过质心到达最低健康高度、碰到大腿以上作为是否倾倒依据）、get_body_com是MujocoEnv中的一种方法，获取“base_link”的质心位置
    def is_healthy(self):
        min_z = self._healthy_z_range
        # 计算相对高度
        shin_height = min((self.get_body_com("left_shin")[2]-self.get_body_com("left_wheel")[2]), (self.get_body_com("right_shin")[2]-self.get_body_com("left_wheel")[2]))  # 获取小腿的高度
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

    # 执行仿真中的一步
    def step(self, action):
        self.do_simulation(action, self.frame_skip)                             # 更新仿真
        self.action = action
        if (int((self.sim.data.time/0.002)+0.5)) % (50) == 0:                   # 更新控制指令
            self._resample_commands()

        observation = self._get_obs()                                           # 获取观察值
        self.lin_vel_agent_xyz = torch.tensor(self.vel_agent, dtype=torch.float32).to(self.device)                  # 获取Agent的线速度
        self.lin_vel_agent_local = self.lin_vel_agent_xyz[0]
        # 计算奖励
        r_vel = self._reward_tracking_vel()  
        r_vel_z = self._reward_lin_vel_z()
        r_ang_xy = self._reward_ang_vel_xy()
        r_torque = self._reward_torque()
        r_gravity = self._reward_gravity()
        r_similar_legged = self._reward_similar_legged()
        r_stand = self._reward_nominal_state()
        r_height = self._reward_base_height()
        r_energy = self._reward_energy()
        r_elur = self._reward_base_ang_xy()
        
        reward = 1.3*r_vel + r_vel_z*0.2 + r_torque*0.3 + r_gravity*0.3 + r_ang_xy*0.2 + r_similar_legged*0.3 + r_stand*0.08 + self.healthy_reward*0.5 + r_height*0.1 + r_energy*0.2 + r_elur*0.4                           # 定义奖励函数
        done = self.done                        # 只要没死亡就可以一直仿真

        info = {}
        return observation, reward.tolist(), done, info
    

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
        pos_id = self.model.sensor_name2id("Body_Pos")
        # 获取Agent上传感器imu的速度、角速度和位姿
        self.vel_agent = self.sim.data.sensordata[self.sim.model.sensor_adr[vel_id]:self.sim.model.sensor_adr[vel_id] + self.sim.model.sensor_dim[vel_id]]
        self.gyro_agent= self.sim.data.sensordata[self.sim.model.sensor_adr[gyro_id]:self.sim.model.sensor_adr[gyro_id] + self.sim.model.sensor_dim[gyro_id]]
        self.quat_imu = self.position[3:7]
        # self.acc_agent = self.sim.data.sensordata[self.sim.model.sensor_adr[acc_id]:self.sim.model.sensor_adr[acc_id] + self.sim.model.sensor_dim[acc_id]]
        # 获取重力投影
        self.gravity = torch.tensor([0, 0, 0, -9.81], dtype=torch.float32).to(self.device)
        quat_conj = torch.tensor([self.quat_imu[0], -self.quat_imu[1], -self.quat_imu[2], -self.quat_imu[3]], dtype=torch.float32).to(self.device)
        on = self.quaternion_multiply(self.quat_imu, self.gravity)
        self.gravity_local = self.quaternion_multiply(on, quat_conj)
        observations = np.concatenate((self.position[3:], self.velocity, self.torques,self.vel_agent, self.gyro_agent, self.gravity_local, self.commands))
        return observations

    # 重置模型
    def reset_model(self):
        self._resample_commands()
        observation = self._get_obs()
        return observation

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

        r_square = torch.exp(-lin_vel_error_square) + 1.2 * torch.exp(-ang_vel_error_square)
        r_abs = torch.exp(-lin_vel_error_abs * 2) + 1.2 * torch.exp(-ang_vel_error_abs * 2)
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
        r=torch.sum(torch.square(torch.from_numpy(self.torques)))
        return torch.exp(-r)
    # 低速奖励维持默认动作
    def _reward_nominal_state(self):
        stand_command = (torch.norm(self.commands) <= self.min_stand)
        r = torch.exp(- torch.sum(torch.tensor(self.position[7:9])) - torch.sum(torch.tensor(self.position[10:12])))
        r = torch.where(stand_command, r, torch.zeros_like(r))
        return r

    def _reward_gravity(self):
        # 计算重力对Agent的影响
        r = torch.abs(self.gravity_local[3] - self.gravity[3])
        return torch.exp(-r)

    def _reward_similar_legged(self):
        left_leg = torch.tensor(self.position[7:9])
        right_leg = torch.tensor(self.position[10:12])
        r = torch.sum(torch.square(left_leg - right_leg))
        return torch.exp(-r)

    def _reward_base_height(self):
        r = torch.square(torch.tensor(self.get_body_com("base_link")[2]-self.get_body_com("left_wheel")[2]) - torch.tensor(self.target_height))
        return torch.exp(-r)
    
    def _reward_energy(self):
        r = torch.sum(torch.abs(torch.tensor(self.action)))
        return torch.exp(-r)

    def _reward_base_ang_xy(self):
        pitch, roll =self.get_agent_euler()[1:]
        r = torch.square(pitch) + torch.square(roll)
        return torch.exp(-r*2)
