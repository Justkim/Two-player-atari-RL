#Parts of this example are from: https://github.com/iffiX/machin/blob/master/examples/framework_examples/dqn_per.py
from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
from pettingzoo.atari import space_invaders_v2
import supersuit as ss
import numpy as np
import wandb
import argparse
import copy
import datetime
import os
from pathlib import Path

env = space_invaders_v2.parallel_env(obs_type='ram')
env = ss.frame_skip_v0(env, 4)
# # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
env = ss.dtype_v0(env, np.dtype("float64"))
env = ss.normalize_obs_v0(env)
env = ss.clip_reward_v0(env)

# configurations
observe_dim = 128
action_num = 6
max_steps = 200
solved_reward = 190
solved_repeat = 5
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

def change_agent(obs_input):
    obs = np.copy(obs_input)
    temp = obs_input[29]
    obs[29] = obs_input[28]
    obs[28] = temp
    return obs
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="space_invaders_v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--transfer-path", type=str, default="")
    parser.add_argument("--self-play-step", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default='')
    parser.add_argument("--episode", type=int, default=20)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--random-opponent", action="store_true", default=False)
    parser.add_argument("--self-play", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # setting the seed for both numpy and torch
    np.random.seed(args.seed)
    t.manual_seed(args.seed)

    max_episodes = args.episode


    if args.transfer_path != '':
    #     print("No transfer path in self-play mode, abort!")
    #     exit()
        transfer_path = args.transfer_path

        transfer_model = t.load(transfer_path, map_location=args.device)
        transfer_model_modified = {}
        transfer_model_copy = copy.deepcopy(transfer_model)
        for key in transfer_model.keys():
            if 'model' and 'fc' in key:
                pre, middle, post = key.split('.')
                transfer_model_modified[middle+"."+post] = transfer_model_copy.pop(key)
        transfer_model_modified['fc3.weight'] = transfer_model_copy.pop('model.Q.0.weight')
        transfer_model_modified['fc3.bias'] = transfer_model_copy.pop('model.Q.0.bias')
        print("transferred bits: ", transfer_model_modified.keys())
    if args.wandb:
        wandb.init(project="machin_transfer", entity="justkim42")

    q_net = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    q_net_t = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    if args.transfer_path != '':
        q_net.load_state_dict(transfer_model_modified)

    opponent_q_net = QNet(observe_dim, action_num, args.device, args.device).double().to(args.device)
    opponent_q_net.eval()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "dqn_per"
    log_name = os.path.join("spaceInvaders", args.algo_name, str(args.seed), now)
    log_path = os.path.join('.', log_name)
    Path(log_path).mkdir(parents=True, exist_ok=True) 
    dqn_per = DQNPer(q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"), batch_size = 256)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    total_step = 0
    episode_len = 0
    total_reward = 0
    terminal = False
    observations, infos = env.reset(seed=args.seed)
    state = t.tensor(observations['first_0'], dtype=t.float64)
    tmp_observations = []

    while episode < max_episodes:
        if episode % save_step == 0:
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
                    action2 = int(opponent_q_net.forward(opponent_observation).argmax().cpu())
                actions = {'first_0':action1_cpu, 'second_0':action2}
                # take an step
                observations, rewards, terminations, truncations, infos = env.step(actions)
                total_step += 1
                episode_len += 1
                state = t.tensor(observations['second_0'], dtype=t.float64)
                total_reward += rewards['first_0']

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
                wandb.log({"agent reward": rewards['first_0'], "timestep": total_step})
                wandb.log({"opponent reward": rewards['second_0'], "timestep": total_step})
            terminal = terminations['first_0'] or truncations['first_0']
        #Things that should happen at the end of the episode
        episode += 1
                # update, update more if episode is longer, else less
        if episode > 20:
            for _ in range(episode_len):
                dqn_per.update()
        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} smoothed total reward={smoothed_total_reward:.2f}")
        wandb.log({"total_reward": total_reward, "episode": episode})
        wandb.log({"total_smoothed_reward": smoothed_total_reward, "episode": episode})
        wandb.log({"episode len": episode_len, "episode": episode})
        dqn_per.store_episode(tmp_observations)
        total_reward = 0
        episode_len = 0
    
        observations, infos = env.reset(seed=args.seed)
        state = t.tensor(observations['first_0'], dtype=t.float64)
        tmp_observations = []

    
    t.save(dqn_per.qnet.state_dict(), os.path.join(log_path, "final_policy.pth"))
    t.save(q_net.state_dict(), os.path.join(log_path, "final_policy.pth"))
    t.save({
            'epoch': max_episodes,
            'model_state_dict': dqn_per.qnet.state_dict(),
            'optimizer_state_dict': dqn_per.qnet_optim.state_dict(),
            },  os.path.join(log_path, "checkpoint"))




