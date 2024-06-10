#Parts of this example are from: https://github.com/iffiX/machin/blob/master/examples/framework_examples/dqn_per.py
from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger
import logging
from pathlib import Path
import torch as t
import torch.nn as nn
import gym
from pettingzoo.atari import space_invaders_v2, pong_v3, boxing_v2, tennis_v3, surround_v2
import supersuit as ss
import numpy as np
import wandb
import argparse
import copy
import datetime
import os
import random
from pathlib import Path
import pickle
from simhash import SimhashIndex

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="space_invaders")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--transfer-path", type=str, default="")
    parser.add_argument("--self-play-step", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default='')
    parser.add_argument("--episode", type=int, default=20)
    parser.add_argument("--clip-rewards", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--random-opponent", action="store_true", default=False)
    parser.add_argument("--self-play", action="store_true", default=False)
    parser.add_argument("--epsilon", type=float, default=0.9999985)
    parser.add_argument("--opponent-randomness", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--transfer", action="store_true", default=False)
    parser.add_argument("--freeze-first-layer", action="store_true", default=False)
    parser.add_argument("--freeze-two-layer", action="store_true", default=False)
    return parser.parse_args()

def log_args():
    logger.info("task: {}".format(args.task))
    logger.info("device: {}".format(args.device))
    logger.info("transfer_path: {}".format(args.transfer_path))
    logger.info("self-play-step: {}".format(args.self_play_step))
    logger.info("seed: {}".format(args.seed))
    logger.info("log-path: {}".format(args.log_path))
    logger.info("transfer: {}".format(args.transfer))
    logger.info("episode: {}".format(args.episode))
    logger.info("wandb: {}".format(args.wandb))
    logger.info("random-opponent: {}".format(args.random_opponent))
    logger.info("self-play: {}".format(args.self_play))
    logger.info("epsilon: {}".format(args.epsilon))
    logger.info("opponent-randomness: {}".format(args.opponent_randomness))
    logger.info("clip-rewards: {}".format(args.clip_rewards))
    logger.info("freeze-first-layer: {}".format(args.freeze_first_layer))
    logger.info("freeze-two-layer: {}".format(args.freeze_two_layer))


args = get_args()

if args.task == "pong":
    env = pong_v3.parallel_env(obs_type='ram')
elif args.task == "space_invaders":
    env = space_invaders_v2.parallel_env(obs_type='ram')
elif args.task == "boxing":
    env = boxing_v2.parallel_env(obs_type='ram')
elif args.task == "tennis":
    env = tennis_v3.parallel_env(obs_type='ram')
elif args.task == "surround":
    env = surround_v2.parallel_env(obs_type='ram')
else:
    logger.error("Environment not found!")
    exit()

env = ss.frame_skip_v0(env, 4)
# # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.dtype_v0(env, np.dtype("float64"))
env = ss.normalize_obs_v0(env)
if args.clip_rewards:
    env = ss.clip_reward_v0(env)
# configurations
observe_dim = 128 #always this number if you work with ram
action_num = env.action_space('first_0').n
logger.info("action_num: {}".format(action_num))
max_steps = 200
save_step = 5000




# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num, input_device, output_device):
        super().__init__()
        self.double()
        self.input_device = input_device
        self.output_device = output_device
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_num)
        

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)
    
    def freeze_first_layer(self):
        for param in self.fc1.parameters():
                param.requires_grad = False
        logger.info("Layer fc1 is frozen now.")
    def freeze_second_layer(self):
        for param in self.fc2.parameters():
                param.requires_grad = False
        logger.info("Layer fc2 is frozen now.")

def change_agent(obs_input):

    obs = np.copy(obs_input)
    if args.task == "space_invaders":
        temp = obs_input[29]
        obs[29] = obs_input[28]
        obs[28] = temp

        temp = obs_input[105]
        obs[105] = obs_input[104]
        obs[104] = temp

        temp = obs_input[74]
        obs[74] = obs_input[73]
        obs[73] = temp

    elif args.task == "pong":
        temp = obs_input[51]
        obs[51] = obs_input[50]
        obs[50] = temp

        temp = obs_input[46]
        obs[46] = obs_input[45]
        obs[45] = temp

        temp = obs_input[14]
        obs[14] = obs_input[13]
        obs[13] = temp

    elif args.task == "boxing":
        temp = obs_input[33]
        obs[33] = obs_input[32]
        obs[32] = temp

        temp = obs_input[35]
        obs[35] = obs_input[34]
        obs[34] = temp

        temp = obs_input[19]
        obs[19] = obs_input[18]
        obs[18] = temp

    
    elif args.task == "tennis":
        temp = obs_input[27]
        obs[27] = obs_input[26]
        obs[26] = temp

        temp = obs_input[25]
        obs[25] = obs_input[24]
        obs[24] = temp

        temp = obs_input[70]
        obs[70] = obs_input[69]
        obs[69] = temp

    elif args.task == "surround":
        temp = obs_input[110]
        obs[110] = obs_input[109]
        obs[109] = temp

        temp = obs_input[112]
        obs[112] = obs_input[111]
        obs[111] = temp

        temp = obs_input[119]
        obs[119] = obs_input[118]
        obs[118] = temp
    else:
        logger.error("Could not find the environment specifications.")
        exit()
    return obs



if __name__ == "__main__":
    args = get_args()
    wandb_config = args.__dict__
    log_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    if args.transfer:
        log_name = os.path.join(args.task, 'dqn_per', 'transfer', str(args.seed), now)
    else:
        log_name = os.path.join(args.task, 'dqn_per', str(args.seed), now)
    # setting the seed for both numpy and torch
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    random.seed(args.seed)

    max_episodes = args.episode
    if not args.random_opponent and not args.self_play:
       logger.error("The opponent mode is not provided (self-play or random?)")
       exit()

    if args.random_opponent and args.self_play:
       logger.error("The opponent mode is not provided correctly (self-play or random?)")
       exit()

    if args.transfer:
        if args.transfer_path != '':            
            transfer_path = args.transfer_path

            transfer_model = t.load(transfer_path, map_location=args.device)
            transfer_model_modified = {}
            transfer_model_copy = copy.deepcopy(transfer_model)
            for key in transfer_model.keys():
                if 'model' and 'fc' in key and not 'old' in key:
                    pre, middle, post = key.split('.')
                    transfer_model_modified[middle+"."+post] = transfer_model_copy.pop(key)
            transfer_model_modified['fc3.weight'] = transfer_model_copy.pop('model.Q.0.weight')
            transfer_model_modified['fc3.bias'] = transfer_model_copy.pop('model.Q.0.bias')
            print("transferred bits: ", transfer_model_modified.keys())
            assert np.array_equal(transfer_model['model.fc1.weight'], transfer_model_modified['fc1.weight'])
            assert np.array_equal(transfer_model['model.fc1.bias'], transfer_model_modified['fc1.bias'])

            assert np.array_equal(transfer_model['model.fc2.weight'], transfer_model_modified['fc2.weight'])
            assert np.array_equal(transfer_model['model.fc2.bias'], transfer_model_modified['fc2.bias'])

            assert np.array_equal(transfer_model['model.Q.0.weight'], transfer_model_modified['fc3.weight'])
            assert np.array_equal(transfer_model['model.Q.0.bias'], transfer_model_modified['fc3.bias'])
        else:
            logger.error("No transfer path provided.")
            exit()
    if args.wandb:
        wandb_name = log_name.replace('/', '-')
        wandb.init(project="machin_transfer", entity="justkim42", name=wandb_name, config=wandb_config,     settings=wandb.Settings(
        log_internal=str(Path(__file__).parent / 'wandb' / 'null'),
    ))

    q_net = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    q_net_t = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    if args.freeze_two_layer:
        q_net.freeze_first_layer()
        q_net_t.freeze_first_layer()
        q_net.freeze_second_layer()
        q_net_t.freeze_second_layer()
    elif args.freeze_first_layer:
        q_net.freeze_first_layer()
        q_net_t.freeze_first_layer()

    if args.transfer_path != '':
        q_net.load_state_dict(transfer_model_modified)
        q_net_t.load_state_dict(transfer_model_modified)
        logger.info("Transfer done")

    opponent_q_net = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    opponent_q_net.eval()

    log_path = os.path.join('.', log_name)
    Path(log_path).mkdir(parents=True, exist_ok=True) 
    dqn_per = DQNPer(q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"), batch_size = args.batch_size, epsilon_decay=args.epsilon)
    episode, step = 0, 0
    total_step = 0
    episode_len = 0
    total_reward = 0
    total_opponent_reward = 0
    terminal = False
    observations, infos = env.reset(seed=args.seed)
    state = t.tensor(observations['first_0'], dtype=t.float64)
    observation = observations['first_0']
    tmp_observations = []

    while episode < max_episodes:
        if episode % save_step == 0:
            logger.info("Save checkpoint")
            t.save(q_net.state_dict(), os.path.join(log_path, "current_policy.pth"))
            t.save({
                    'epoch': episode,
                    'model_state_dict': dqn_per.qnet.state_dict(),
                    'optimizer_state_dict': dqn_per.qnet_optim.state_dict(),
                    },  os.path.join(log_path, "checkpoint"))
        terminal = False

        while not terminal and episode_len < max_steps:
            if total_step % args.self_play_step == 0:
                #self-play update
                logger.info("Self-play update")
                opponent_q_net.load_state_dict(q_net.state_dict())
            with t.no_grad():
                old_state = state
                # agent model inference
                action1 = dqn_per.act_discrete_with_noise({"state": old_state.view(1, observe_dim)})
                action1_cpu = action1.cpu().numpy()[0][0]
                if args.random_opponent:
                    action2 = env.action_space('second_0').sample()
                elif args.self_play:
                    opponent_observation = t.tensor(change_agent(observations['second_0']), dtype=t.float64).to(args.device)
                    random_number = np.random.rand()
                    if random_number > args.opponent_randomness:
                        action2 = int(opponent_q_net.forward(opponent_observation).argmax().cpu())
                    else:
                        random_number = random.randint(0, action_num-1)
                        action2 = random_number
                actions = {'first_0':action1_cpu, 'second_0':action2}
                # take an step
                observations, rewards, terminations, truncations, infos = env.step(actions)
                total_step += 1
                episode_len += 1
                state = t.tensor(observations['first_0'], dtype=t.float64)
                total_reward += rewards['first_0']
                total_opponent_reward += rewards['second_0']

                experience =  {
                        "state": {"state": old_state.view(1, observe_dim)},
                        "action": {"action": action1},
                        "next_state": {"state": state.view(1, observe_dim)},
                        "reward": rewards['first_0'],
                        "terminal": terminations['first_0'],
                    }
                tmp_observations.append(
                    experience
                )
                if args.wandb:
                    wandb.log({"agent reward": rewards['first_0'], "action": action1_cpu, "timestep": total_step})
                    wandb.log({"opponent reward": rewards['second_0'], "opponent_action": action2, "timestep": total_step})
            terminal = terminations['first_0'] or truncations['first_0']
        #Things that should happen at the end of the episode
        dqn_per.store_episode(tmp_observations)

        if episode > 20:
            for _ in range(episode_len):
                dqn_per.update()

                # update, update more if episode is longer, else less

        # show reward
        #logger.info(f"Episode {episode} reward={total_reward:.2f}")
        if args.wandb:
            wandb.log({"total_reward": total_reward, "episode": episode})
            wandb.log({"total_opponent_reward": total_opponent_reward, "episode": episode})
            wandb.log({"episode len": episode_len, "episode": episode})

       
        total_reward = 0
        total_opponent_reward = 0
        episode_len = 0
    
        observations, infos = env.reset(seed=args.seed)
        state = t.tensor(observations['first_0'], dtype=t.float64)
        tmp_observations = []
        episode += 1

    
    t.save(q_net.state_dict(), os.path.join(log_path, "final_policy.pth"))
    t.save({
            'epoch': max_episodes,
            'model_state_dict': dqn_per.qnet.state_dict(),
            'optimizer_state_dict': dqn_per.qnet_optim.state_dict(),
            },  os.path.join(log_path, "checkpoint"))




