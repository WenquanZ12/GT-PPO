import numpy as np

# 替换为你的 .npy 文件路径
file_path = r'C:\Users\zwq\Desktop\GT-PPO\GT-PPO\Data\generatedData6_6_Seed200.npy'

# 读取 .npy 文件1
data = np.load(file_path)

# 打印数组的基本信息和部分内容
print(f"数组形状: {data.shape}")
print(f"数组数据类型: {data.dtype}")
print("数组部分数据（前5个元素）:")
print(data[:5])
