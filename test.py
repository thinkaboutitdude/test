import numpy as np
import os
import matplotlib.pyplot as plt

returns = [0]
for file in os.listdir('./trajectories/ucb'):
    name = os.path.join('./trajectories/ucb', file, 'trajectories_100.npz')
    npz_file = np.load(name)
    for reward in npz_file['rewards']:
        returns.append(returns[-1] + reward[0])
    print(returns)
    break
plt.plot(np.array(returns))
plt.show()