import numpy as np

# 设置参数
j = 20  # 作业数
m = 14  # 工件数
l = 1   # 最小处理时间
h = 99  # 最大处理时间
batch_size = 100
seed = 200

# 设置随机种子
np.random.seed(seed)

def permute_rows(x):
    '''
    对每一行进行独立随机排列
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]

def uni_instance_gen1(n_j, n_m, low, high):


    b = np.random.randint(low=low, high=high, size=(n_j, n_m))         # 中值
    # 计算上下边界
    delta = (b * 0.3).astype(int)  # 计算 20% 的偏移量

    # 为确保 a < b < c，生成 a 和 c
    a = np.maximum(b - delta, low)  # 左值：确保 a 不小于 low
    c = np.minimum(b + delta, high)  # 右值：确保 c 不大于 high


    # 确保 a, b, c 不相等
    a = np.where(a == b, a - 1, a)  # 如果 a == b，强制修改 a
    c = np.where(c == b, c + 1, c)  # 如果 c == b，强制修改 c


    fuzzy_times = np.stack([a, b, c], axis=2)

    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)

    return fuzzy_times, machines

# 批量生成数据
fuzzy_times_list = []
machines_list = []

for _ in range(batch_size):
    fuzzy_times, machines = uni_instance_gen1(n_j=j, n_m=m, low=l, high=h)
    fuzzy_times_list.append(fuzzy_times)
    machines_list.append(machines)

# 转为 numpy 数组
fuzzy_times_array = np.stack(fuzzy_times_list)   # shape: (batch_size, j, m, 3)
machines_array = np.stack(machines_list)         # shape: (batch_size, j, m)

# 保存为 .npz 文件
np.savez(f'generatedData{j}_{m}_Seed{seed}',
         fuzzy_times=fuzzy_times_array,
         machines=machines_array)

print("数据生成并保存成功！")

