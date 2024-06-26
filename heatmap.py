import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse

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
parser.add_argument("--clip", default=False, action='store_true')
args = parser.parse_args()

data_loaded = np.load('ram_datasets/{}_player_dataset_{}.npz'.format('one',args.task_name))
one_player_dataset_loaded = data_loaded['dataset']


# for i in tqdm(range(100000)):
#     print(one_player_dataset_loaded[i])

errors = average_calc(one_player_dataset_loaded, one_player_dataset_loaded.shape[0], kernel_size=100)

# data_loaded = np.load('two_player_dataset.npz')
# two_player_dataset_loaded = data_loaded['big_dataset']
errors = np.reshape(errors, (32, 4))
# tan_errors = np.tan(errors)
# log_error = np.emath.logn(100, errors)
clip_errors = np.clip(errors, a_min = 0, a_max = 500)

print(one_player_dataset_loaded[21])

data_loaded = np.load('ram_datasets/{}_player_dataset_{}.npz'.format('two',args.task_name))
two_player_dataset_loaded = data_loaded['dataset']
print(two_player_dataset_loaded[21])
print("sdfsdsdfsdfsef")

# for i in tqdm(range(100000)):
#     print(one_player_dataset_loaded[i])

errors2 = average_calc(two_player_dataset_loaded, two_player_dataset_loaded.shape[0], kernel_size=100)

# data_loaded = np.load('two_player_dataset.npz')
# two_player_dataset_loaded = data_loaded['big_dataset']
errors2 = np.reshape(errors2, (32, 4))
# tan_errors = np.tan(errors)
# log_error = np.emath.logn(100, errors)
if args.clip:
    ferrors = np.clip(np.abs(errors - errors2), a_min = 0, a_max = 500)
else:
    ferrors = np.abs(errors - errors2)
    
print(errors)
print(errors2)
print("--------")
print(np.abs(errors - errors2))
print(args.task_name + ": " + str(np.sum(np.abs(errors - errors2))/128))
plt.imshow(ferrors, cmap='hot', interpolation='nearest')
plt.savefig("heatmaps/" + args.task_name + str(args.clip) + "_heatmap_difference.png")
# plt.show()
