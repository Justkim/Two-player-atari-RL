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
from pettingzoo.atari import pong_v3, double_dunk_v3, flag_capture_v2, entombed_cooperative_v3, entombed_competitive_v3, tennis_v3, space_invaders_v2, mario_bros_v3, surround_v2, boxing_v2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    np.random.seed(99)
    random.seed(99)
    parser.add_argument("--num-player", type=int, default=1)
    parser.add_argument("--task-name", type=str, default='')
    args = parser.parse_args()
    if args.num_player == 1:
        if args.task_name == "pong":
            env1 = gym.make('Pong-ramNoFrameskip-v4')
        elif args.task_name == "space_invaders":
            env1 = gym.make('SpaceInvaders-ramNoFrameskip-v4')
        elif args.task_name == "mario_bros":
            env1 = gym.make('ALE/MarioBros-ram-v5')
        elif args.task_name == "surround":
            env1 = gym.make('ALE/Surround-ram-v5')
        elif args.task_name == "tennis":
            env1 = gym.make('Tennis-ramNoFrameskip-v4')
        elif args.task_name == "double_dunk":
            env1 = gym.make('DoubleDunk-ramNoFrameskip-v4')
        elif args.task_name == "entombed":
            env1 = gym.make('ALE/Entombed-ram-v5')
        elif args.task_name == "boxing":
            env1 = gym.make('Boxing-ramNoFrameskip-v4')
        elif args.task_name == "flag_capture":
            env1 = gym.make('ALE/FlagCapture-ram-v5')



        else:
            exit()
    elif args.num_player == 2:
        if args.task_name == "pong":
            env2 = pong_v3.parallel_env(obs_type='ram')
        elif args.task_name == "space_invaders":
            env2 = space_invaders_v2.parallel_env(obs_type='ram')
        elif args.task_name == "boxing":
            env2 = boxing_v2.parallel_env(obs_type='ram')
        elif args.task_name == "tennis":
            env2 = tennis_v3.parallel_env(obs_type='ram')
        elif args.task_name == "surround":
            env2 = surround_v2.parallel_env(obs_type='ram')
        elif args.task_name == "mario_bros":
            env2 = mario_bros_v3.parallel_env(obs_type='ram')
        elif args.task_name == "flag_capture":
            env2 = flag_capture_v2.parallel_env(obs_type='ram', full_action_space=True)
        elif args.task_name == "entombed_competitive":
            env2 = entombed_competitive_v3.parallel_env(obs_type='ram')
        elif args.task_name == "entombed_cooperative":
            env2 = entombed_cooperative_v3.parallel_env(obs_type='ram')
        elif args.task_name == "double_dunk":
            env2 = double_dunk_v3.parallel_env(obs_type='ram')
        
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