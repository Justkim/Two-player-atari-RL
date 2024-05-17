import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
df_2kkuioyc = pd.read_csv('bcu51q0d.csv')
df_myyouc7n =  pd.read_csv('myyouc7n.csv')

moving_avg_2kkuioyc = df_2kkuioyc['total_reward'].rolling(window=20).mean()
moving_avg_myyouc7n = df_myyouc7n['total_reward'].rolling(window=20).mean()

smoothed_2kkuioyc = savgol_filter(df_2kkuioyc['total_reward'], window_length=20, polyorder=2)
#smoothed_y6kjzffd = savgol_filter(df_y6kjzffd['total_reward'], window_length=20, polyorder=2)
smoothed_myyouc7n = savgol_filter(df_myyouc7n['total_reward'], window_length=20, polyorder=2)

# Plotting the time series of given dataframe
plt.plot(df_2kkuioyc.episode, smoothed_2kkuioyc, label='Transferred Space invaders')
#plt.plot(df_y6kjzffd.episode, smoothed_y6kjzffd, label='Line 2')
plt.plot(df_myyouc7n.episode, smoothed_myyouc7n, label='Space Invaders from scratch')

# Giving title to the chart using plt.title
plt.title('Classes by Date')
plt.legend()
# rotating the x-axis tick labels at 30degree 
# towards right
plt.xticks(rotation=30, ha='right')
 
# Providing x and y label to the chart
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.show()