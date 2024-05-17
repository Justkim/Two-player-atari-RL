import pandas as pd
import wandb
import pandas
import csv
api = wandb.Api(timeout=29)

data_list = []

# run is specified by <entity>/<project>/<run_id>
run = api.run("justkim42/machin_transfer/bcu51q0d")
history = run.scan_history(keys=['episode', 'total_reward'], page_size=5000)
# history = run.history(keys=['episode', 'total_reward'], pandas=True)
# history.to_csv("y6kjzffd_sample.csv")
with open('bcu51q0d.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['episode', 'total_reward'])
    for entry in history:
        print(entry)
        data_list.append(entry)
        writer.writerow([entry['episode'], entry['total_reward']])
