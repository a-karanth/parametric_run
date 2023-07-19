# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:02:40 2023

@author: 20181270
"""
import os
import pandas as pd
import numpy as np

directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\\PhD\\TRNSYS\\Publication1\\pub_1\\src\\res'
trn_results = '\\trn'
os.chdir(directory)
# df = pd.read_csv('1.txt', delimiter='\t', header=None)
# df2=df.transpose()
# df2.columns = df2.iloc[0]
# df2 = df2.drop(df2.index[0])

# df2.to_csv('1.txt', index=False)

# check original sample files
df1 = pd.read_csv('morris_st_sample.csv')
df2 = pd.read_csv('morris_pvt_sample.csv')

# adding the parameter in the old sample files
df1['r_level'] = 'r0'
df2['r_level'] = 'r0'
df1.to_csv('morris_st_sample.csv', index=False)
df2.to_csv('morris_pvt_sample.csv', index=False)

og = pd.concat([df1,df2])
og = og.drop_duplicates()
og.index = np.arange(len(og))
og.dtypes #check the datatypes of all columns for comparison later
#%%
os.chdir(directory+trn_results)

df = pd.read_csv('list_of_inputs.csv',header=0)

# adding a new column.
# df['r_level'] = 'r0'
# df.to_csv('list_of_inputs.csv', index=False)
#%% combine all .txt files into 1 file
# df_ip = pd.DataFrame(columns=['flow_rate','volume','coll_area','design_case'])
# for i in np.arange(0,86):
#     df = pd.read_csv(str(i)+'.txt', delimiter=',', header=0)
#     df_ip.loc[i] = df.iloc[0]
# df_ip = df_ip[['volume', 'coll_area', 'flow_rate', 'design_case']]

# df_ip['volume'] = df_ip['volume'].astype('float64')
# df_ip['coll_area'] = df_ip['coll_area'].astype('int64')
# df_ip['flow_rate'] = df_ip['flow_rate'].astype('int64')

# if df_ip.equals(og) == True:
#     df_ip.to_csv('list_of_inputs.csv', index=False)
#%% transpose all .txt files to make the data row data. and add a column for design case
# for i in np.arange(0,86):
#     df = pd.read_csv(str(i)+'.txt', delimiter='\t', header=None)
#     df2=df.transpose()
#     df2.columns = df2.iloc[0]
#     df2 = df2.drop(df2.index[0])
#     df2['desing_case'] = 'PVT'
#     df2.to_csv(str(i)+'.txt', index=False)

#%% rename desing_case in all files. and change area to coll_area
# for i in np.arange(0,86):
#     df = pd.read_csv(str(i)+'.txt', delimiter=',', header=0)
#     df.rename(columns = {'area':'coll_area'}, inplace = True)
#     df.rename(columns = {'desing_case':'design_case'}, inplace = True)
#     df.to_csv(str(i)+'.txt', index=False)

