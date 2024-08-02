# The initial version of this code is written by: Yiwei Zhang
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import supersuit as ss
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
from atari_wrapper import make_atari_env



def reset(env):
    if args.task == "space_invaders":
        observations, infos = env.reset(seed=99)
        for i in range(130):
            actions = {'first_0': 0, 'second_0': 0}
            observations, rewards, terminations, truncations, infos = env.step(actions)
    elif args.task == "pong":
        observations, infos = env.reset(seed=99)
        for i in range(60):
            actions = {'first_0': 0, 'second_0': 0}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        
    else:
        observations, infos = env.reset(seed=99)
    return observations, infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    np.random.seed(99)
    random.seed(99)
    parser.add_argument("--num-player", type=int, default=1)
    parser.add_argument("--task", type=str, default='')
    args = parser.parse_args()
    if args.num_player == 1:
        if args.task == "pong":
            env1 = 'Pong-ramNoFrameskip-v4'
        elif args.task == "space_invaders":
            env1 = 'SpaceInvaders-ramNoFrameskip-v4'
        elif args.task == "mario_bros":
            env1 = 'ALE/MarioBros-ram-v5'
        elif args.task == "surround":
            env1 = 'ALE/Surround-ram-v5'
        elif args.task == "tennis":
            env1 = 'Tennis-ramNoFrameskip-v4'
        elif args.task == "double_dunk":
            env1 = 'DoubleDunk-ramNoFrameskip-v4'
        elif args.task == "entombed":
            env1 ='ALE/Entombed-ram-v5'
        elif args.task == "boxing":
            env1 = 'Boxing-ramNoFrameskip-v4'
        elif args.task == "flag_capture":
            env1 = 'ALE/FlagCapture-ram-v5'
        


        else:
            exit()

        env1, train_envs, test_envs = make_atari_env(
        env1,
        99,
        1,
        1,
        scale=False,
        frame_stack=False,
        mode='ram'
    
    )


    elif args.num_player == 2:
        if args.task == "pong":
            env2 = pong_v3.parallel_env(obs_type='ram')
        elif args.task == "space_invaders":
            env2 = space_invaders_v2.parallel_env(obs_type='ram')
        elif args.task == "boxing":
            env2 = boxing_v2.parallel_env(obs_type='ram')
        elif args.task == "tennis":
            env2 = tennis_v3.parallel_env(obs_type='ram')
        elif args.task == "surround":
            env2 = surround_v2.parallel_env(obs_type='ram')
        elif args.task == "mario_bros":
            env2 = mario_bros_v3.parallel_env(obs_type='ram')
        elif args.task == "flag_capture":
            env2 = flag_capture_v2.parallel_env(obs_type='ram', full_action_space=True)
        elif args.task == "entombed_competitive":
            env2 = entombed_competitive_v3.parallel_env(obs_type='ram')
        elif args.task == "entombed_cooperative":
            env2 = entombed_cooperative_v3.parallel_env(obs_type='ram')
        elif args.task == "double_dunk":
            env2 = double_dunk_v3.parallel_env(obs_type='ram')
        else:
            exit()
        env2 = ss.frame_skip_v0(env2, 4)
        # # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
        env2 = ss.sticky_actions_v0(env2, repeat_action_probability=0.25)
        # env2 = AgentIndicatorAtariEnv(env2)
        # env2 = ss.dtype_v0(env2, np.dtype("float64"))
        # env2 = ss.normalize_obs_v0(env2)
        # env2 = ss.clip_reward_v0(env2)



    args = parser.parse_args()

    batch_size = 50000
    batch = []
    if args.num_player == 1:
        env1.reset(seed=99)
    else:
        reset(env2)
    inner_step = 0
    for step in tqdm(range(batch_size)):
        if args.num_player == 1:
            action1 = env1.action_space.sample()
            observation1, rew1, done, _, _ = env1.step(action1)
            inner_step += 1
            if done:
                env1.reset(seed=99)
                inner_step = 0
            batch.append(observation1)
        elif args.num_player == 2:
            action2 =   {'first_0': env2.action_space('first_0').sample(), 'second_0': env2.action_space('second_0').sample()}
            observation2, rew2, term, trunc, _ = env2.step(action2)
            observation2 = observation2["first_0"]
            inner_step += 1
            if term['first_0'] or trunc['first_0'] or inner_step >= 200:
                reset(env2)
                inner_step = 0
            batch.append(observation2)
    batch_array = np.array(batch)
    if args.num_player == 1:
        np.savez('../ram_datasets/one_player_dataset_{}.npz'.format(args.task), dataset=batch_array)
    elif args.num_player == 2:
        np.savez('../ram_datasets/two_player_dataset_{}.npz'.format(args.task), dataset=batch_array)