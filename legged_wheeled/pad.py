import pygame
import numpy as np
from pathlib import Path
class control_gamepad:
    def __init__(self,command_cfg,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        self.use_gamepad = True
        self.running = True
        screen_width = 800
        screen_height = 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("This use your keyboard")
        image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
        image_surface.fill((255, 255, 255, 0))  # 透明初始化

        
        # 获取连接的游戏手柄数量
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad,open keyboard window")
            self.use_gamepad = False
            pygame.quit()
            exit()
        else:
            # 选择第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = command_cfg["num_commands"]
        self.command_cfg = command_cfg
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 1.0 ,0.05]
    
    
    def get_commands(self):
        pygame.event.pump()
        # 重置
        reset_flag = False
        if self.use_gamepad:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    # print(f"按钮 {event.button} 被按下。")
                    if event.button == 0:
                        reset_flag=True
                # elif event.type == pygame.JOYBUTTONUP:
                #     print(f"按钮 {event.button} 被释放。")
                elif event.type == pygame.JOYAXISMOTION:
                    # print(f"轴 {event.axis},{event.value}")
                    if event.axis== 1: #ly 前正后负
                        self.commands[0] = -event.value * self.command_scale[0]
                    elif event.axis== 2: #rx 左正右负
                        self.commands[1] = -event.value * self.command_scale[1]
        else:
            quiet_walking = 1.0
            for event in pygame.event.get():  # 获取事件队列中的所有事件
                if event.type == pygame.QUIT:  # 用户点击窗口关闭按钮
                    self.running = False
                elif event.type == pygame.KEYDOWN:  # 键盘按键按下事件
                    if event.key == pygame.K_w:
                        self.commands[0] = self.command_scale[0] * quiet_walking
                    elif event.key == pygame.K_s:
                        self.commands[0] = -self.command_scale[0] * quiet_walking
                    elif event.key == pygame.K_a:
                        self.commands[1] = self.command_scale[1] * quiet_walking
                    elif event.key == pygame.K_d:
                        self.commands[1] = -self.command_scale[1] * quiet_walking
                elif event.type == pygame.KEYUP:  # 键盘按键释放事件
                    if event.key == pygame.K_w:
                        self.commands[0] = 0
                    elif event.key == pygame.K_s:
                        self.commands[0] = 0
                    elif event.key == pygame.K_a:
                        self.commands[1] = 0
                    elif event.key == pygame.K_d:
                        self.commands[1] = 0
        if self.running == False:
            pygame.quit()
            exit()
        self.commands_clip()
        return self.commands,reset_flag
    #将速度范围限制在
    def commands_clip(self):
        # lin_vel_x
        if self.commands[0] < self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][0] * self.command_scale[0]
        elif self.commands[0] > self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]:
            self.commands[0] = self.command_cfg["lin_vel_x_range"][1] * self.command_scale[0]

        #ang_vel
        if self.commands[1] < self.command_cfg["ang_vel_range"][0] * self.command_scale[2]:
            self.commands[1] = self.command_cfg["ang_vel_range"][0] * self.command_scale[2]
        elif self.commands[1] > self.command_cfg["ang_vel_range"][1] * self.command_scale[2]:
            self.commands[1] = self.command_cfg["ang_vel_range"][1] * self.command_scale[2]
