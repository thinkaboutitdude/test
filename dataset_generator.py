from dataclasses import dataclass, asdict
import pyrallis
from typing import Tuple, Optional
from collections import defaultdict
import uuid
import os 
import json
import shutil
import multiprocessing as mp

import numpy as np
import gymnasium as gym
import wandb
import env

@dataclass
class Config:
    # wandb params
    project: str = "ICRL_AD"
    group: str = "UCB_BANDITS"
    name: Optional[str] = None

    # seeds
    train_seed: int = 0
    eval_seed: int = 100

    # data settings
    env_name: str = "MultiArmedBanditBernoulli"
    num_arms: int = 10
    num_train_envs: int = 10_000
    num_env_steps: int = 100
    num_in_context_steps: int = 100
    ucb_alpha: float = 0.3  # ucb
    learning_histories_path: Optional[str] = "trajectories/ucb"

    num_eval_envs: int = 200
    eval_every: int = 500
    log_every: int = 100

    num_train_steps: int = 50_000
    rand_select_in_ucb: bool = True


def skewed_reward_dist(rng: np.random.Generator, max_arms: int, num_envs: int):
    """
    This function creates a set of bandits which have the higher rewards distributed over the even arms.
    """
    num_even_arms = max_arms // 2
    num_odd_arms = max_arms - num_even_arms
    means = np.zeros((num_envs, max_arms))
    # 95% of the time - even arms have higher return
    means[:, ::2] = rng.uniform(size=(num_envs, num_odd_arms), low=0.0, high=0.5)
    means[:, 1::2] = rng.uniform(
        size=(num_envs, num_even_arms),
        low=0.5,
        high=1.0,
    )

    return means


def mixed_skewed_reward_dist(
    rng: np.random.Generator, max_arms: int, num_envs: int, frac_first: float
):
    """
    This function creates two sets of bandits. The first one contains bandits with the higher rewards
    distributed over the odd arms. The second set is similar but with the even arms.
    :param max_arms: controls the maximum amount of arms in all bandits
    :param num_envs: the total amount of envs in two sets combined
    :param frac_first: the relative size of the first set compared to the num_envs
    """
    offset = int(num_envs * frac_first)

    # create the first 'odd' set
    means1 = skewed_reward_dist(rng, max_arms=max_arms, num_envs=offset)
    # create the second 'even' set
    means2 = skewed_reward_dist(rng, max_arms=max_arms, num_envs=num_envs - offset)
    means2 = 1 - means2

    # check that the bandits in the sets are correct
    assert means1[0, 0] < means1[0, 1], (means1[0, 0], means1[0, 1])
    assert means2[0, 0] > means2[0, 1], (means2[0, 0], means2[0, 1])

    # combine the two sets
    means = np.concatenate([means1, means2], axis=0)
    # shuffle the bandits
    means = rng.permutation(means)

    return means

def make_envs(
    config: Config,
) -> Tuple[np.ndarray]:
    """
    This function will create arm distribution for train and test envs,
    a single one for each episode.
    Instead of putting this logic in the environment itself,
    I do it in the beginning of the program for a better control over
    the kinds of distributions and the relationships between train and test distributions
    I want to get. Also for reproducibility.
    """

    # Higher rewards are more likely to be distributed under
    # odd arms 95% of the time during training
    rng = np.random.default_rng(config.train_seed)

    train_means = mixed_skewed_reward_dist(
        rng,
        max_arms=config.num_arms,
        num_envs=config.num_train_envs,
        frac_first=0.95,
    )

    rng = np.random.default_rng(config.eval_seed)
    # Higher rewards are more likely to be distributed under
    # even arms 95% of the time during eval
    eval_means = mixed_skewed_reward_dist(
        rng,
        max_arms=config.num_arms,
        num_envs=config.num_eval_envs,
        frac_first=0.05,
    )
    
    return (train_means, eval_means)
    

# If not doing it in this way, have to deal with warnings
def calc_delta(alpha, num_pulled, t):
    if num_pulled > 0:
        return np.sqrt(alpha * np.log(t) / num_pulled)
    else:
        return 1e20


def calc_av_reward(sum_rewards, num_pulled):
    if num_pulled > 0:
        return sum_rewards / num_pulled
    else:
        return 0


calc_delta = np.vectorize(calc_delta)
calc_av_reward = np.vectorize(calc_av_reward)

class UCB:
    def __init__(self, alpha: float, num_arms: int, rand_select: bool):
        self.num_arms = num_arms

        self.num_pulled = np.zeros(num_arms)
        self.sum_rewards = np.zeros(num_arms)
        self.t = 0
        self.alpha = alpha

        self.rand_select = rand_select

    def select_arm(self):
        delta = calc_delta(self.alpha, self.num_pulled, self.t)
        av_reward = calc_av_reward(self.sum_rewards, self.num_pulled)

        value = av_reward + delta
        if self.rand_select:
            max_arm_value = np.max(value)
            max_arms = np.argwhere(value == max_arm_value).flatten()
            one_max_arm = np.random.choice(max_arms)
        else:
            one_max_arm = np.argmax(value)

        return one_max_arm

    def update_state(self, arm, reward):
        self.num_pulled[arm] += 1
        self.sum_rewards[arm] += reward
        self.t += 1

class Random:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def select_arm(self):
        act = np.random.randint(low=0, high=self.num_arms)

        return act

    def update_state(self, arm, reward):
        pass

def save_metadata(savedir, save_filenames):
    metadata = {
        "algorithm": "LinUCB",
        "label": "label",
        "ordered_trajectories": save_filenames,
    }
    with open(os.path.join(savedir, "metadata.metadata"), mode="w") as f:
        json.dump(metadata, f, indent=2)


def dump_trajs(savedir: str, idx: int, trajs: list):
    filename = os.path.join(savedir, f"trajectories_{idx}.npz")
    np.savez(
        filename,
        states=np.array(trajs["states"], dtype=np.float32).reshape(-1, 1),
        actions=np.array(trajs["actions"], dtype=np.int32).reshape(-1, 1),
        rewards=np.array(trajs["rewards"], dtype=np.float32).reshape(-1, 1),
        dones=np.array(
            np.array(trajs["terminateds"]) | np.array(trajs["truncateds"]),
            dtype=np.int32,
        ).reshape(-1, 1),
        num_actions=np.array(trajs["num_actions"], dtype=np.float32),
    )
    return os.path.basename(filename)

def calc_all_actions_i(actions, num_actions):
    """
    Calculates the step when all actions are tried
    """
    acc = np.zeros(num_actions)
    for i, a in enumerate(actions):
        acc[a] += 1
        if np.all(acc > 0):
            return i

    return np.sum(acc > 0)


def solve_bandit(
    env: gym.Env, algo: UCB, savedir: Optional[str], max_steps: int, seed: int
):
    trajs = defaultdict(list)

    state, _ = env.reset(seed=seed)
    regrets = []
    alphas = []
    for _ in range(max_steps):
        action = algo.select_arm()
        new_state, reward, term, trunc, info = env.step(action)
        assert not (term or trunc)

        regrets.append(info["regret"])

        algo.update_state(action, reward)

        # save transitions
        trajs["states"].append(state)
        trajs["actions"].append(action)
        trajs["rewards"].append(reward)
        trajs["terminateds"].append(term)
        trajs["truncateds"].append(trunc)

        state = new_state

        if hasattr(algo, "alpha"):
            alphas.append(algo.alpha)

    alphas = np.array(alphas)

    # fraction of steps when the optimal action was used
    frac_optimal = np.mean(np.array(trajs["actions"]) == info["opt_act"])

    trajs["num_actions"].append(algo.num_arms)
    if savedir is not None:
        filename = dump_trajs(savedir, max_steps, trajs)
        save_metadata(savedir=savedir, save_filenames=[os.path.basename(filename)])

    # record the step index when all actions are tried at least once
    all_actions_i = calc_all_actions_i(trajs["actions"], env.unwrapped.action_space.n)
    return (
        np.array(regrets),
        np.array(alphas),
        np.array(trajs["actions"]),
        frac_optimal,
        all_actions_i,
    )

class Worker:
    """
    This class is used to run the data generation algorithm on a single bandit instance.
    :param data_generation_algo: which algorithm is used for data generation
    :param num_env_steps: number of steps in the environment that an algorithm performs
    :param basedir: where to save the data
    :param seed: a random seed
    """

    def __init__(
        self,
        config: Config,
        data_generation_algo: str,
        num_env_steps: int,
        basedir: str,
        seed: int,
    ):
        self.config = config
        self.basedir = basedir
        self.num_env_steps = num_env_steps
        self.data_generation_algo = data_generation_algo
        self.seed = seed

    def __call__(self, inp):
        means = inp
        # Create environment
        env = gym.make(self.config.env_name, arms_mean=means, num_arms=self.config.num_arms)
        # Create a random name for this history's logs
        id = uuid.uuid4()
        if self.basedir is not None:
            savedir = os.path.join(self.basedir, f"LinUCB-{id}")
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = None

        # Choose a data generation algorithm
        if self.data_generation_algo == "ucb":
            algo = UCB(
                alpha=self.config.ucb_alpha,
                num_arms=self.config.num_arms,
                rand_select=self.config.rand_select_in_ucb,
            )
        elif self.data_generation_algo == "random":
            algo = Random(num_arms=self.config.num_arms)
        else:
            raise NotImplementedError

        # Run the data generation algorithm
        regrets, alphas, actions, frac_optimal, all_actions_i = solve_bandit(
            env=env,
            algo=algo,
            savedir=savedir,
            max_steps=self.num_env_steps,
            seed=self.seed,
        )

        return regrets, alphas, actions, frac_optimal, all_actions_i
    
def solve_bandits(
    config: Config,
    arms_means: np.ndarray,
    basedir: str,
    data_generation_algo: str,
    num_env_steps: int,
    seed: int,
):
    """
    Run the data generation algorithm on each bandit and get the metrics back.
    """

    # Generate trajectories
    with mp.Pool(processes=os.cpu_count()) as pool:
        out = pool.map(
            Worker(
                config=config,
                basedir=basedir,
                data_generation_algo=data_generation_algo,
                num_env_steps=num_env_steps,
                seed=seed,
            ),
            arms_means,
        )
        regrets, alphas, actions, frac_optimal, all_actions_i = zip(*out)

    regrets = np.asarray(regrets)
    actions = np.asarray(actions)
    alphas = np.asarray(alphas)
    frac_optimal = np.asarray(frac_optimal)
    all_actions_i = np.asarray(all_actions_i)

    return regrets, alphas, actions, frac_optimal, all_actions_i


def generate_dataset(
    config: Config, arms_means: np.ndarray, seed: int
):
    regrets, alphas, actions, frac_optimal, all_actions_i = solve_bandits(
        config=config,
        arms_means=arms_means,
        basedir=config.learning_histories_path,
        data_generation_algo="ucb",
        num_env_steps=config.num_env_steps,
        seed=seed,
    )

def main(config: Config):
    # wandb.init(
    #     project=config.project,
    #     group=config.group,
    #     name=config.name,
    #     config=asdict(config),
    #     save_code=True,
    # )

    # create the bandits for train and test
    train_means, eval_means = make_envs(config=config)

    # Run the data generation algorithm and log the training histories
    generate_dataset(
        config=config,
        arms_means=train_means,
        seed=config.train_seed,
    )

if __name__ == '__main__':
    config = pyrallis.parse(config_class=Config)
    main(config)