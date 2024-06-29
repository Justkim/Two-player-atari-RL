import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse


def load(dataset_name):
    data_loaded = np.load('ram_datasets/{}_player_dataset_{}.npz'.format(dataset_name,args.task_name))
    dataset_loaded = data_loaded['dataset']
    return dataset_loaded

def calculate_temporal_variance(dataset_loaded):

    variance = average_calc(dataset_loaded, dataset_loaded.shape[0], kernel_size=11)
    variance = np.reshape(variance, (32, 4))
    return variance

def single_byte_average_calc(data, x, kernel_size):
    sum = 0
    # for y in range(kernel_size):
    #     sum += data[x + y][i]
    # return sum / kernel_size
    sum_values = np.sum(data[x:x+kernel_size], axis=0)
    return sum_values / kernel_size


def average_calc(data, data_size, kernel_size=3):  # kernel_size is assumed to be odd

    diff_sum = np.zeros(shape=data.shape[1])

    for x in range(data_size - kernel_size + 1):
        # print("x: ", x)
        # print("[x][i]: ", data[x][i])
        average = single_byte_average_calc(data, x, kernel_size)
        # print("average: ", average)
        diff = abs(data[x + (int)((kernel_size - 1) / 2)] - average)
        # print("diff: ", diff)
        diff = np.square(diff)
        diff_sum += diff
    return diff_sum / (data_size - kernel_size + 1)
parser = argparse.ArgumentParser()
parser.add_argument("--task-name", type=str, default='')
parser.add_argument("--num-player", type=int, default=0)
parser.add_argument("--clip", default=False, action='store_true')

args = parser.parse_args()

if args.num_player < 2:
    one_player_dataset_loaded = load('one')
    one_variance = calculate_temporal_variance(one_player_dataset_loaded)

if args.num_player  == 0 or args.num_player == 2:
    two_player_dataset_loaded = load('two')
    two_variance = calculate_temporal_variance(two_player_dataset_loaded)

if args.num_player == 0:
    if args.clip:
        ferrors = np.clip(np.abs(one_variance - two_variance), a_min = 0, a_max = 500)
    else:
        ferrors = np.abs(one_variance - two_variance)
    plt.imshow(ferrors, cmap='hot')
    print(args.num_player, args.task_name, np.sum(ferrors)/128)
    plt.savefig("heatmaps/" + args.task_name + str(args.clip) + "_error_heatmap.png")
if args.num_player == 1:
    plt.imshow(one_variance, cmap='hot')
    print(args.num_player, args.task_name, np.sum(one_variance)/128)
    plt.savefig("heatmaps/" + args.task_name + "_ram_one_heatmap.png")
if args.num_player == 2:
    print("-------------------")
    print(args.num_player, args.task_name, np.sum(two_variance)/128)
    # two_variance = (two_variance-np.min(two_variance))/(np.max(two_variance)-np.min(two_variance)) 
    # print(two_variance)
    plt.imshow(two_variance, cmap='hot')
    plt.savefig("heatmaps/" + args.task_name + "_ram_two_heatmap.png")
