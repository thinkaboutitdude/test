import torch
import torch.nn as nn
import pyrallis
import wandb
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from functools import partial
from dataclasses import dataclass, asdict
from data.dataset import SequenceData
from torch.utils.data import DataLoader
from algos.transformer.model import Model, ModelConfig
from algos.transformer.schedule import cosine_annealing_with_warmup
from tqdm import trange
from torch.nn.utils import clip_grad_norm_
from utils.set_seed import set_seed
from algos.random import Random
from typing import List
from gymnasium.vector import SyncVectorEnv
from dataset_generator import UCBConfig, generate_envs, generate_trajectories

@dataclass
class TrainConfig:
    """
    Config for train transformer
    """
    train_seed: int = 1
    eval_seed: int = 0
    num_arms: int = 10
    num_train_steps: int = 10000
    seq_len: int = 100
    num_episodes = 20
    episode_steps = 100
    eval_every: int = 1000
    layer_norm_bias: bool = True
    token_embed_dim: int = 128
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 1e-4
    beta1: float = 0.9
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.1
    clip_grad_norm: float = 5
    get_action_type: str = "sample"
    label_smoothing: float = 0.02

    device: str = "cuda"
    histories_path: str = 'trajectories.npz'
    checkpoint_path: str = 'checkpoints'

def make_env(env: gym.Env):
    def helper():
        return env
    return helper


def make_input_for_eval(
    actions: torch.Tensor,
    rewards: torch.Tensor,
    istep: int,
):

    if istep == 0:
        num_envs = actions.shape[1]
        inp = (
            torch.empty(
                num_envs,
                0,
                dtype=actions.dtype,
                device=actions.device,
            ),
            torch.empty(num_envs, 0, dtype=rewards.dtype, device=rewards.device),
        )
    else:
        inp = (actions.transpose(0, 1)[:, -istep:], rewards.T[:, -istep:])

    return inp

def eval_random(ucbconfig: UCBConfig):
    """
    Here we try to make random agent for future comparisons
    """
    _, eval_envs = generate_envs(ucbconfig)
    algo = Random(ucbconfig.num_arms)
    _, mean_returns = generate_trajectories(eval_envs, algo, ucbconfig, mode='eval')
    return mean_returns

@torch.no_grad()
def evaluate_in_context(
    config: TrainConfig,
    ucbconfig: UCBConfig,
    model: Model
):
    """
    Evaluation function for checking in context learning
    """
    _, eval_envs = generate_envs(ucbconfig)
    vec_env = SyncVectorEnv([make_env(env) for env in eval_envs])
    
    vec_env.reset(seed=config.eval_seed)
    actions = torch.zeros(
        (config.seq_len, len(eval_envs)),
        dtype=torch.long,
        device=config.device,
    )
    rewards = torch.zeros(
        (config.seq_len, len(eval_envs)), dtype=torch.long, device=config.device
    )
    mean_returns = [0]
    returns = np.zeros(vec_env.num_envs)

    for istep in trange(config.seq_len, desc="Eval ..."):
        sliced_actions, sliced_rewards = make_input_for_eval(
            actions=actions,
            rewards=rewards,
            istep=istep,
        )
        # check for validity
        assert (istep < config.seq_len and sliced_actions.shape[1] == istep) or (
            istep >= config.seq_len and sliced_actions.shape[1] == config.seq_len
        ), (
            sliced_actions.shape[1],
            istep,
        )
        # make prediction
        pred = model(actions=sliced_actions, rewards=sliced_rewards)
        pred = pred[:, -1]
        dist = torch.distributions.Categorical(logits=pred)
        action_sample = dist.sample()
        action_mode = pred.argmax(dim=-1)
        if config.get_action_type == "sample":
            action = action_sample
        elif config.get_action_type == "mode":
            action = action_mode
        action = action.squeeze(-1)
        _, reward, _, _, info = vec_env.step(action.cpu().numpy())
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        actions[-1] = action
        rewards[-1] = torch.from_numpy(reward).type(torch.long).to(config.device)
        returns = reward
        mean_returns.append(mean_returns[-1] + returns.mean())

    return np.array(mean_returns)


def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch

def train(config: TrainConfig, ucbconfig: UCBConfig):
    """
    This is main train function
    """
    set_seed(seed=config.train_seed)
    wandb.init(project='ad_rl_2', config=asdict(config))
    dataset = SequenceData(data_path=config.histories_path, context_len=config.seq_len)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    model_config = ModelConfig(
        n_embd=config.d_model, 
        n_layer=config.num_layers,
        n_query_head=config.num_heads,
        attn_pdrop=config.attention_dropout, 
        resid_pdrop=config.dropout, 
        block_size=2 * config.seq_len + 2, 
        rope=True
    )

    model = Model(model_config, n_token=config.token_embed_dim, num_actions=config.num_arms).to(config.device)
    model.apply(partial(model._init_weights, n_layer=model_config.n_layer))

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, 0.999),
    )

    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=int(config.num_train_steps * config.warmup_ratio),
        total_steps=config.num_train_steps,
    )

    model.train()

    dataloader = next_dataloader(dataloader)

    for global_step in trange(1, config.num_train_steps + 1, desc="Training"):
        _, actions, rewards = next(dataloader)
        actions = actions.to(torch.long).to(config.device)
        rewards = rewards.to(torch.long).to(config.device)
        assert actions.shape[1] == config.seq_len, (
                actions.shape[1],
                config.seq_len,
            )
        assert rewards.shape[1] == config.seq_len, (
            rewards.shape[1],
            config.seq_len,
        )
        pred = model(actions, rewards)
        pred = pred[:, :-1]
        assert pred.shape[1] == config.seq_len, (pred.shape[1], config.seq_len)
        loss = torch.nn.functional.cross_entropy(
            input=pred.flatten(0, 1),
            target=actions.flatten(0, 1),
            label_smoothing=config.label_smoothing,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optim.step()
        scheduler.step()
        wandb.log(
            {
                'step': global_step, 
                'loss': loss.item()
            }
        )

        if global_step % config.eval_every == 0:
            model.eval()

            mean_returns = evaluate_in_context(config=config, ucbconfig=ucbconfig, model=model)
            plt.plot(mean_returns)
            wandb.log(
                {
                    f'returns_sum_ad_{global_step}': plt, 
                }
            )

            mean_returns_random = eval_random(ucbconfig)
            plt.plot(mean_returns_random)
            wandb.log(
                {
                    f'returns_sum_random_{global_step}': plt
                }
            )
            model.train()

    
    checkpoint_full_path = os.path.join(config.checkpoint_path, 'model_checkpoint.pt')
    os.makedirs(os.path.dirname(checkpoint_full_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_full_path)

if __name__ == '__main__':
    config = TrainConfig()
    ucbconfig = UCBConfig()
    train(config=config, ucbconfig=ucbconfig)