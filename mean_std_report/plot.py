import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import os
import seaborn as sns
from numpy import trapz
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize

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

merged_data = pd.concat(list_of_non_transfer, axis=1, ignore_index=False)
# Commented sampling
# print(original_merged_data.shape)
# original_merged_data.to_csv('test.csv')
# merged_data = original_merged_data.sample(n=10000, axis=0, random_state = 99)
# print(merged_data.head)
# print(merged_data.shape)
# merged_data = merged_data.sort_index()
print(merged_data.shape)
merged_data['rolling_mean'] = merged_data.mean(axis=1)
merged_data['rolling_std']  = merged_data.std(axis=1)


# plt.figure(figsize=(10, 6))
# # # Plot the mean
# plt.plot(merged_data['rolling_mean'], label='Non transfer Mean')

# # Plot the standard deviation
# plt.plot(row_stds, label='Standard Deviation', marker='o')
# plt.fill_between(merged_data.index, merged_data['rolling_mean'] - merged_data['rolling_std'], merged_data['rolling_mean'] + merged_data['rolling_std'], 
#                  color='red', alpha=0.3, label='Rolling Std Dev')

merged_data_t = pd.concat(list_of_transfer, axis=1, ignore_index=False)
print(merged_data_t.shape)

# merged_data_t = original_merged_data_t.sample(n=10000, axis=0, random_state = 99)
# merged_data_t = merged_data_t.sort_index()
# # Step 3: Plot the mean and standard deviation
# # rolling(window=20
merged_data_t['rolling_mean'] = merged_data_t.mean(axis=1)
merged_data_t['rolling_std']  = merged_data_t.std(axis=1)
merged_data.to_csv('test1.csv')
merged_data_t.to_csv('test2.csv')

print("non-transfer")
print('0:', merged_data['rolling_mean'][0], '+', merged_data['rolling_std'][0])
print('5000:', merged_data['rolling_mean'][5000-1], '+', merged_data['rolling_std'][5000-1])
print('10000:', merged_data['rolling_mean'][10000-1], '+', merged_data['rolling_std'][10000-1])
print('15000:', merged_data['rolling_mean'][15000-1], '+', merged_data['rolling_std'][15000-1])
print('20000:', merged_data['rolling_mean'][20000-1], '+', merged_data['rolling_std'][20000-1])
print("--------------------------------------")
print("transfer")
print('0:', merged_data_t['rolling_mean'][0], '+', merged_data_t['rolling_std'][0])
print('5000:', merged_data_t['rolling_mean'][5000-1], '+', merged_data_t['rolling_std'][5000-1])
print('10000:', merged_data_t['rolling_mean'][10000-1], '+', merged_data_t['rolling_std'][10000-1])
print('15000:', merged_data_t['rolling_mean'][15000-1], '+', merged_data_t['rolling_std'][15000-1])
print('20000:', merged_data_t['rolling_mean'][20000-1], '+', merged_data_t['rolling_std'][20000-1])
print("--------------------------------------")
print("Last 10 final episode avarage")
no_transfer_avg = merged_data['rolling_mean'][-10:].mean()
transfer_avg = merged_data_t['rolling_mean'][-10:].mean()
print("no-transfer:", no_transfer_avg)
print("transfer:", transfer_avg)
print("percentage change last 10",(transfer_avg - no_transfer_avg) / abs(no_transfer_avg)* 100)
print("/////////////////////")
no_transfer_avg = merged_data['rolling_mean'][-100:].mean()
transfer_avg = merged_data_t['rolling_mean'][-100:].mean()
print("no-transfer:", no_transfer_avg)
print("transfer:", transfer_avg)
print("percentage change last 100",(transfer_avg - no_transfer_avg) / abs(no_transfer_avg)* 100)
print("/////////////////////")
min_val = -1.008
max_val = 12.903999999999998
norm_no_transfer = (no_transfer_avg - min_val) / (max_val - min_val)
norm_transfer = (transfer_avg - min_val) / (max_val - min_val)
diff = norm_transfer - norm_no_transfer
print("old_norm_no_transfer:", norm_no_transfer)
print("old_norm_transfer:", norm_transfer)
print("old_diff", diff)
print("//////////////////////")
non_transfer_area = trapz(merged_data['rolling_mean'], dx=10)
print("non transfer area =", non_transfer_area)
transfer_area = trapz(merged_data_t['rolling_mean'], dx=10)
print("transfer area =", transfer_area)
print("percentage change area",(transfer_area - non_transfer_area) / abs(non_transfer_area)* 100)
#print("percentage diff",(transfer_avg - no_transfer_avg) / (abs(transfer_avg + no_transfer_avg) / 2))
print("##########################################")
# merged_data.to_csv('merged_data_pandas.csv')
# merged_data_t.to_csv('merged_data_pandas.csv')
# mean_merged_data = merged_data[['rolling_mean']]
# mean_merged_data = mean_merged_data.sort_values(by='rolling_mean', ignore_index=True)
# print("mean merged data")
# print(mean_merged_data['rolling_mean'][19999])
# Q1 = mean_merged_data.quantile(0.25, axis=0)
# Q3 = mean_merged_data.quantile(0.75, axis=0)
# IQR = Q3 - Q1
# threshold = 1.5
# filtered_merged_data = mean_merged_data[(mean_merged_data < Q1 - threshold * IQR) | (mean_merged_data > Q3 + threshold * IQR)]
# print("imp", filtered_merged_data.shape)
# no_transfer_min = filtered_merged_data.min(axis=1)
# no_transfer_min.to_csv('merged_data_pandas.csv')
# print(no_transfer_min.shape)
merged_data['rolling_mean'] = winsorize(merged_data['rolling_mean'], limits=[0.05, 0.05])
merged_data_t['rolling_mean'] = winsorize(merged_data_t['rolling_mean'], limits=[0.05, 0.05])
no_transfer_max = merged_data['rolling_mean'].max()
no_transfer_min = merged_data['rolling_mean'].min()
transfer_min = merged_data_t['rolling_mean'].min()
transfer_max = merged_data_t['rolling_mean'].max()
print(transfer_min, no_transfer_min)
print(transfer_max, no_transfer_max)
min_val = min(transfer_min, no_transfer_min)
max_val = max(transfer_max, no_transfer_max)
no_transfer_norm = ((merged_data['rolling_mean'][-100:]- min_val) /(max_val - min_val)).mean() 
transfer_norm = ((merged_data_t['rolling_mean'][-100:].mean() - min_val) /(max_val - min_val)).mean() 
print("new_norm_no_transfer:", no_transfer_norm)
print("new_norm_transfer:", transfer_norm)
print("new_diff", transfer_norm - no_transfer_norm)
print("##########################################")
# # # Plot the mean
plt.plot(merged_data_t['rolling_mean'], label='Transfer Mean')
plt.plot(merged_data['rolling_mean'], label='Mean')
# # # Plot the standard deviation
# # plt.plot(row_stds, label='Standard Deviation', marker='o')
# plt.fill_between(merged_data_t.index, merged_data_t['rolling_mean'] - merged_data_t['rolling_std'], merged_data_t['rolling_mean'] + merged_data_t['rolling_std'], 
#                  color='orange', alpha=0.3, label='Rolling Std Dev')

# # Adding titles and labels
# plt.title('Row-wise Mean and Standard Deviation')
# plt.xlabel('Row Index')
# plt.ylabel('Value')
# plt.legend()

# Show the plot
# plt.show()


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