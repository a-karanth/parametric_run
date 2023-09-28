import pandas as pd
import numpy as np

# Sample DataFrame
data = {'start': [1, 3, 5],
        'end': [10, 20, 30]}
data = pd.read_csv('AEF.csv', index_col=0)
data = data.drop(columns=['Annual_average','Unnamed: 4','Unnamed: 5'])
data.plot()
#%%
df = pd.DataFrame(data)

# Number of interpolation points (columns to be added between start and end)
num_interpolations = 5

# Calculate the step size
step_size = (df['July'] - df['January']) / (num_interpolations + 1)

# Create equally spaced values between start and end
interpolated_values = [df['January'] + step_size * (i + 1) for i in range(num_interpolations)]

# Create DataFrame for interpolated values and concatenate with the original DataFrame
interpolated_df = pd.DataFrame(interpolated_values).transpose()
interpolated_df.columns = [f'interp_{i}' for i in range(1, num_interpolations + 1)]
df = pd.concat([df, interpolated_df], axis=1)
df.plot()

#%%
df = df.rename(columns={"January": 1, "July": 7, 'interp_1':2,
                   'interp_2':3, 'interp_3':4, 'interp_4':5, 'interp_5':6})
#%%
df[8] = df[6]
df[9] = df[5]
df[10] = df[4]
df[11] = df[3]
df[12] = df[2]

#%% 
df.to_csv('AEF_annual.csv',index=True)