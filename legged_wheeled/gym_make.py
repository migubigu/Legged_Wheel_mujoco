import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# 设置高度场的分辨率
width, height = 300, 300

# 创建随机高度场
terrain = np.random.rand(height, width) * 40

# 使用高斯滤波平滑地形
terrain = gaussian_filter(terrain, sigma=1.2)

# 将高度场归一化到 0-200
terrain = ((terrain - terrain.min()) / (terrain.max() - terrain.min()) * 40).astype(np.uint8)

# 保存为 PNG 文件
image = Image.fromarray(terrain, mode='L')
image.save("./envs/asset/png/terrain.png")