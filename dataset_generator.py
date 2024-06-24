# The initial version of this code is written by: Yiwei Zhang
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import gymnasium as gym
from datetime import datetime
import argparse
# Define the Autoencoder model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.atari import pong_v3
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-player", type=int, default=1)
    parser.add_argument("--task-name", type=str, default='')
    args = parser.parse_args()
    if args.num_player == 1:
        if args.task_name == "pong":
            env1 = gym.make('Pong-ramNoFrameskip-v4')
        else:
            exit()
    elif args.num_player == 2:
        if args.task_name == "pong":
            env2 = pong_v3.parallel_env(obs_type='ram')
        else:
            exit()


    args = parser.parse_args()

    batch_size = 5000
    batch = []
    if args.num_player == 1:
        env1.reset(seed=99)
    else:
        env2.reset(seed=99)

    for step in tqdm(range(batch_size)):
        if args.num_player == 1:
            action1 = env1.action_space.sample()
            observation1, rew1, done, _, _ = env1.step(action1)
            if done:
                env1.reset(seed=99)
            batch.append(observation1)
        elif args.num_player == 2:
            action2 =   {'first_0': env2.action_space('first_0').sample(), 'second_0': env2.action_space('second_0').sample()}
            observation2, rew2, term, trunc, _ = env2.step(action2)
            observation2 = observation2["first_0"]
            if term or trunc:
                env2.reset(seed=99)
            batch.append(observation2)
    batch_array = np.array(batch)
    if args.num_player == 1:
        np.savez('ram_datasets/one_player_dataset_{}.npz'.format(args.task_name), dataset=batch_array)
    elif args.num_player == 2:
        np.savez('ram_datasets/two_player_dataset_{}.npz'.format(args.task_name), dataset=batch_array)
    # data_loaded = np.load('one_player_dataset_{}.npz'.format(args.task_name))
    # large_board_dataset_loaded = data_loaded['big_dataset']
    # print(large_board_dataset_loaded.shape)