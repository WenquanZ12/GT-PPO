import numpy as np


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]



def uni_instance_gen1(n_j, n_m, low, high):

    b = np.random.randint(low=low, high=high, size=(n_j, n_m))         # 中值

    delta = (b * 0.3).astype(int)  # 计算 30% 的偏移量

    a = np.maximum(b - delta, low)  # 左值：确保 a 不小于 low
    c = np.minimum(b + delta, high)  # 右值：确保 c 不大于 high


    # 确保 a, b, c 不相等
    a = np.where(a == b, a - 1, a)  # 如果 a == b，强制修改 a
    c = np.where(c == b, c + 1, c)  # 如果 c == b，强制修改 c


    fuzzy_times = np.stack([a, b, c], axis=2)

    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)

    return fuzzy_times, machines

def override(fn):
    """
    override decorator
    """
    return fn


