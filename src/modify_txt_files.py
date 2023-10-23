# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:02:40 2023

@author: 20181270
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PostprocessFunctions import PostprocessFunctions as pf

directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\\PhD\\TRNSYS\\Publication1\\pub_1\\src\\res'
trn_results = '\\trn'
# os.chdir(directory)

#%% open and change all csv files in res_folder
allfiles = os.listdir()
csv_files = list(filter(lambda f: f.endswith('.csv'), allfiles))
csv_files = csv_files[:-1]
for file in csv_files:
    df = pd.read_csv(file)
    df['design_case'] = df['design_case'].replace(['PVT','PVT_Batt_6','PVT_Batt_9'],['PVT_0','PVT_6','PVT_9'])
    df.to_csv(file, index=False)

#%% check original sample files
df1 = pd.read_csv('morris_st_sample.csv')
df2 = pd.read_csv('morris_pvt_sample.csv')

# adding the parameter in the old sample files
df1['r_level'] = 'r0'
df2['r_level'] = 'r0'
# df1.to_csv('morris_st_sample.csv', index=False)
# df2.to_csv('morris_pvt_sample.csv', index=False)

og = pd.concat([df1,df2])
og = og.drop_duplicates(ignore_index=True)
og.dtypes #check the datatypes of all columns for comparison later

#%% Modify all results files: remove tabs, delete unused columns and save the files again
directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\\PhD\\TRNSYS\\Publication1\\pub_1\\src\\'
res_folder = 'res\\'
trn_folder = 'res\\trn\\'

temp = os.listdir(directory+trn_folder)
labels = []
for i in temp:
    if '_temp_flow' in i:
        prefix = i[:-14]
    elif '_control_signal' in i:
        prefix = i[:-19]
    elif '_energy' in i:
        prefix = i[:-11]
    else:
        continue
    labels.append(prefix)

labels = np.array(labels)
labels = np.unique(labels)
check_labels = np.array([''.join(filter(str.isdigit, s)) for s in labels])

#%%% check if sim_results.csv exists. create if it doesnt, add new values to it, if it does
sim_yn =  os.listdir(directory+res_folder)
if 'sim_results.csv' in sim_yn:
    existing_res = pd.read_csv(directory+res_folder + 'sim_results.csv',index_col='label')
    existing_labels = np.array(existing_res.index.astype(str).tolist())
    new_labels = list(set(check_labels)-set(existing_labels)) # newly simulated labels
    labels = [i for i in labels if any(j in i for j in new_labels)] #checks labels that exist, and copies the exact name (including _cp) for all new labels
else:
    labels = labels
t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

#%%% read and modify files
def read_modify_files(label):
    
    prefix = directory + trn_folder + label
    temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter=",",index_col=0)
    energy = pd.read_csv(prefix+'_energy.txt', delimiter=",", index_col=0)
    controls = pd.read_csv(prefix+'_control_signal.txt', delimiter=",",index_col=0)
    
    temp_flow.columns = [col.strip() for col in temp_flow.columns]
    energy.columns = [col.strip() for col in energy.columns]
    controls.columns = [col.strip() for col in controls.columns]
    
    temp_flow.drop(columns=['T2_dhw','T3_dhw','T4_dhw','T5_dhw',
                            'T2_sh','T3_sh','T4_sh','T5_sh',
                            'Taux2tap', 'maux2tap',], 
                   inplace=True, errors='ignore')
    energy.drop(columns=['Qaux_tap',], inplace=True, errors='ignore')
    
    # temp_flow.to_csv(prefix+'_temp_flow.txt')
    # energy.to_csv(prefix+'_energy.txt')
    # controls.to_csv(prefix+'_control_signal.txt')

#%%% run parallely using Joblib
from joblib import Parallel, delayed
import time
num_processes = 8  # Change this to the desired number of processes
t1 = time.time()
Parallel(n_jobs=num_processes)(delayed(read_modify_files)(label) for label in labels)
t2 = time.time()
print(t2-t1)
#%% add new columns to list_of inputs file
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

#%% add column to sim_results file
res = pd.read_csv('sim_results.csv', index_col='label')
res = res.rename(columns={'el_bill': 'el_bill_1'})
#%% to extract net metering percentage
nm = ''.join([i for i in test if i.isdigit() or i=='.'])