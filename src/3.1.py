#  PIL 是 Python Imaging Library 的缩写，是 Python 官方的图像处理库
from PIL import Image
import numpy as np
import torch

# 1. 读取图片，将此路径替换为你的图片路径
image = Image.open('./data/img.png')

# 2. 将图片转换为numpy的ndarray格式
image_array = np.array(image)
print("Numpy ndarray格式的图片数据：")
print("数据类型：", image_array.dtype)
print("形状：", image_array.shape)

# 3. 将numpy的ndarray转换为PyTorch的tensor格式
image_tensor = torch.from_numpy(image_array)
print("PyTorch tensor格式的图片数据：")
print("数据类型：", image_tensor.dtype)
print("形状：", image_tensor.shape)
