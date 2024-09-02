import gymnasium as gym
import pyrallis
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from typing import List
from env.prepare_envs import make_envs
from algos.ucb import UCB
from utils.set_seed import set_seed


@dataclass
class UCBConfig:
    """
    Config for dataset generation
    """
    env_name: str = 'MultiArmedBanditBernoulli'
    num_train_envs: int = 5000
    num_eval_envs: int = 100
    ucb_alpha: float = 2.0
    num_arms: int = 10
    train_steps: int = 100
    context_len: int = 100
    train_seed: int = 1
    eval_seed: int = 0

def save_trajectories(data: np.ndarray, save_file: str = 'trajectories.npz') -> None:
    np.savez(save_file, data=data)

def generate_envs(config: UCBConfig):
    train_envs = []
    eval_envs = []
    train_means, eval_dists = make_envs(config)
    for i in range(config.num_train_envs):
        env = gym.make(config.env_name, arms_mean=train_means[i], num_arms=config.num_arms)
        train_envs.append(env)
    for i in range(config.num_eval_envs):
        env = gym.make(config.env_name, arms_mean=eval_dists['inverse'][i], num_arms=config.num_arms)
        eval_envs.append(env)
    return (train_envs, eval_envs)


def generate_trajectories(envs: List[gym.Env], algo: UCB, config: UCBConfig, mode='train'):
    total_history = []
    steps = config.train_steps
    if mode == 'eval':
        steps = config.context_len
    returns = np.zeros(len(envs), steps)
    for env_index in range(len(envs)):
        if mode == 'train':
            state, _ = envs[env_index].reset(seed=config.train_seed)
        else:
            state, _ = envs[env_index].reset(seed=config.eval_seed)
        history = []
        for step in range(steps):
            action = algo.select_arm()
            new_state, reward, term, trunc, info = envs[env_index].step(action)
            algo.update_state(action, reward)
            history.extend([state, action, reward])
            returns[env_index, step] += reward
            state = new_state
        total_history.append(history)
    total_history = np.array(total_history)
    mean_returns = [0]
    for step in range(steps):
        mean_returns.append(mean_returns[-1] + returns[:, step].mean())
    return total_history, np.array(mean_returns)

@pyrallis.wrap()
def main(config: UCBConfig):
    set_seed(config.train_seed)
    train_envs, _ = generate_envs(config)
    algo = UCB(config.ucb_alpha, config.num_arms)
    total_history, _ = generate_trajectories(train_envs, algo, config)
    save_trajectories(total_history)

if __name__ == '__main__':
    main()