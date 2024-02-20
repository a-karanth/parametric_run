# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:07:41 2023

@author: 20181270
"""
import os
import pandas as pd

folder = 'C:\\TRNSYS18\\Tess Models\\SampleCatalogData\\Water-to-WaterHeatPumps\\Normalized\\'
cdata = folder + 'WWHP_Cooling-Normalized.dat'
hdata = folder + 'WWHP_Heating-Normalized.dat'

# hdata = 'C:/Users/20181270/Downloads/WWHP_Heating-Normalized_2kW.dat'
with open(hdata, 'r') as file:
    lines = file.readlines()
#%%
data = []
for line in lines:
    if not line.startswith('!'):
        values = line.strip().split('\t')
        data.append([float(value) if (value.replace('.', '', 1).lstrip('-')).isdigit() else value for value in values if value != ''])
        
# temp = data[3]
# element_last = temp[-1].split(' ', 1)
# float_part = float(element_last[0])
# temp[-1] = float_part
# temp.append(element_last[1])
# data[3] = temp
        # temp = []
        # for value in values:
        #     if value.replace('.', '', 1).isdigit():
        #         temp.append(float(value))
        #     elif value != '':
        #         temp.append(value)
        # data.append(temp)

#%%
inputs = {}
for i in range(4):
    key = data[i][-1]
    values = data[i][:-1]
    inputs[key] = values
    
#%%
order_of_keys = ['!Entering Source Temperatures',
                 '!Entering Load Temperatures',
                 '!Normalized Source Flow Rate',
                 '!Normalized Load Flow Rate']
input_rows = []
for load_flow in inputs['!Normalized Load Flow Rates']:
    for source_flow in inputs['!Normalized Source Flow Rates']:
        for t_load in inputs['!Entering Load Temperatures (C)']:
            for t_source in inputs['!Entering Source Temperatures (C)']:
                input_rows.append([t_source, t_load, source_flow, load_flow])
    
#%%
power_data = data[4:] 
power_data = [[j for j in i if not isinstance(j, str)] for i in power_data]
df_data = [row1 + row2 for row1, row2 in zip(input_rows, power_data)]

columns = ['Tevaporator', 'Tload_in','Feva','Fcond','Capacity','Power']

df = pd.DataFrame(data=df_data, columns=columns)   
#%% Input file has capacity, power and flow as normalized values
#   Scaling them to my inputs used in TRNSYS
cap_scale = 30000/3600 #in kW
pow_scale = 6000/3600 # in kW
load_flow_scale = 1400/3600 # in kg/s 
df['Capacity'] = df['Capacity'] * cap_scale
df['Power'] = df['Power'] * pow_scale

df['COP'] = df['Capacity']/df['Power']
df['Fcond'] = df['Fcond'] * load_flow_scale
#%%
df['Tcondensor'] = df['Capacity']*1000/(df['Fcond']* 4186) +  df['Tload_in']
#%%
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(df['Tcondensor'], df['Tevaporator'], df['COP'], c=df['COP'],cmap='viridis')
ax.set_xlabel('Condensor temperature')
ax.set_ylabel('Evaporator temperature')
ax.set_zlabel('COP')