import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import os
import seaborn as sns

directory_path = sys.argv[1]
list_of_transfer = []
list_of_non_transfer = []
for filename in os.listdir(directory_path):
    if 'transfer' in filename:
            print(filename)
            df = pd.read_csv(os.path.join(directory_path, filename))
            list_of_transfer.append(df)
    else:
            print(filename)
            df = pd.read_csv(os.path.join(directory_path, filename))
            list_of_non_transfer.append(df)

original_merged_data = pd.concat(list_of_non_transfer, axis=1, ignore_index=False)
print(original_merged_data.shape)
original_merged_data.to_csv('test.csv')
merged_data = original_merged_data.sample(n=10000, axis=0, random_state = 99)
print(merged_data.head)
print(merged_data.shape)
merged_data = merged_data.sort_index()

merged_data['rolling_mean'] = merged_data.mean(axis=1).rolling(window=100).mean()
merged_data['rolling_std']  = merged_data.std(axis=1).rolling(window=100).mean()


plt.figure(figsize=(10, 6))
# # Plot the mean
plt.plot(merged_data['rolling_mean'], label='Non transfer Mean')

# # Plot the standard deviation
# plt.plot(row_stds, label='Standard Deviation', marker='o')
plt.fill_between(merged_data.index, merged_data['rolling_mean'] - merged_data['rolling_std'], merged_data['rolling_mean'] + merged_data['rolling_std'], 
                 color='red', alpha=0.3, label='Rolling Std Dev')

original_merged_data_t = pd.concat(list_of_transfer, axis=1, ignore_index=False)
print(original_merged_data_t.shape)
original_merged_data_t.to_csv('test.csv')
merged_data_t = original_merged_data_t.sample(n=10000, axis=0, random_state = 99)
merged_data_t = merged_data_t.sort_index()
# # Step 3: Plot the mean and standard deviation
# # rolling(window=20
merged_data_t['rolling_mean'] = merged_data_t.mean(axis=1).rolling(window=100).mean()
merged_data_t['rolling_std']  = merged_data_t.std(axis=1).rolling(window=100).std()
# # Plot the mean
plt.plot(merged_data_t['rolling_mean'], label='Transfer Mean')

# # Plot the standard deviation
# plt.plot(row_stds, label='Standard Deviation', marker='o')
plt.fill_between(merged_data_t.index, merged_data_t['rolling_mean'] - merged_data_t['rolling_std'], merged_data_t['rolling_mean'] + merged_data_t['rolling_std'], 
                 color='orange', alpha=0.3, label='Rolling Std Dev')

# Adding titles and labels
plt.title('Row-wise Mean and Standard Deviation')
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.show()


# print(merged_data)

    
# # Save the merged DataFrame to a new CSV file
# merged_data.to_csv('merged_data_pandas.csv', index=False)

# moving_avg_2kkuioyc = df_2kkuioyc['total_reward'].rolling(window=20).mean()
# moving_avg_myyouc7n = df_myyouc7n['total_reward'].rolling(window=20).mean()

# smoothed_2kkuioyc = savgol_filter(df_2kkuioyc['total_reward'], window_length=20, polyorder=2)
# #smoothed_y6kjzffd = savgol_filter(df_y6kjzffd['total_reward'], window_length=20, polyorder=2)
# smoothed_myyouc7n = savgol_filter(df_myyouc7n['total_reward'], window_length=20, polyorder=2)

# # Plotting the time series of given dataframe
# plt.plot(df_2kkuioyc.episode, smoothed_2kkuioyc, label='Transferred Space invaders')
# #plt.plot(df_y6kjzffd.episode, smoothed_y6kjzffd, label='Line 2')
# plt.plot(df_myyouc7n.episode, smoothed_myyouc7n, label='Space Invaders from scratch')

# # Giving title to the chart using plt.title
# plt.title('Classes by Date')
# plt.legend()
# # rotating the x-axis tick labels at 30degree 
# # towards right
# plt.xticks(rotation=30, ha='right')
 
# # Providing x and y label to the chart
# plt.xlabel('Episode')
# plt.ylabel('Reward')

# plt.show()