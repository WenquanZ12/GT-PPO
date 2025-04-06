import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 20
m = 10
l = 1
h = 99
batch_size = 100
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(data.shape)
np.save('generatedData{}_{}_Seed{}.npy'.format(j, m, seed), data)

import numpy as np


def generate_TFN(b):

    a = np.random.randint(0, b // 2 + 1)  # Random integer in [0, b/2]
    c = np.random.randint(0, b // 2 + 1) +b # Random integer in [0, b/2]
    return (a, b, c)


# Example usage
np.random.seed(42)  # Set seed for reproducibility
b_values = [10, 20, 30, 40, 50]  # Example crisp processing times
tfn_values = [generate_TFN(b) for b in b_values]

for b, tfn in zip(b_values, tfn_values):
    print(f"b = {b} -> TFN: {tfn}")
