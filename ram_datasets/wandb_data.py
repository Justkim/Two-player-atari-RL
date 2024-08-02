import pandas as pd
import wandb
import pandas
import csv
import sys
import os
from tqdm import tqdm
from os import path
api = wandb.Api(timeout=29)

data_list = []
directory_path = sys.argv[1]
run_link = sys.argv[2]
print(run_link)
# run is specified by <entity>/<project>/<run_id>
run = api.run(run_link)
history = run.scan_history(keys=['episode', 'total_reward'], page_size=1000000)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
with open(path.join('/home/kimiya/Projects/Two-player-atari-RL/',directory_path, run.name + ".csv"), 'w', newline='') as file:
    print(run.name)
    writer = csv.writer(file)
    writer.writerow([run.name])
    counter = 0
    for entry in history:
        writer.writerow([entry['total_reward']])
        counter += 1
