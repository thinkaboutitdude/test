# https://github.com/corl-team/ad-eps/blob/main/dark_room/utils/misc.py

import os
import torch
import random
import numpy as np
import time


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_test_goals(grid_size, num_train_goals, seed):
    set_seed(seed)
    assert num_train_goals <= grid_size ** 2

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = np.random.permutation(goals)

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]
    return train_goals, test_goals

def norm_regret(regrets, lower_regrets, upper_regrets):
    normalized_regret = (regrets.mean(0)[-1] - lower_regrets.mean(0)[-1]) / (
        upper_regrets.mean(0)[-1] - lower_regrets.mean(0)[-1]
    )

    return normalized_regret

class Timeit:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.Event(enable_timing=True)
            self.end_gpu = torch.cuda.Event(enable_timing=True)
            self.start_gpu.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            self.end_gpu.record()
            torch.cuda.synchronize()
            self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        else:
            self.elapsed_time_gpu = -1.0
        self.elapsed_time_cpu = time.time() - self.start_cpu