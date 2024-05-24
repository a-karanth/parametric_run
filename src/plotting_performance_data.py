# -*- coding: utf-8 -*-
"""
Plotting the performance map of heat pump

@author: 20181270
"""
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
        '''
        if value.replace('.', '', 1).lstrip('-')).isdigit(): # if the list element is number after removing - sign and decimal, then
            float(value)                                     # make a float of the value
        else:                                                # or leave it as the value itself   
            value                                            # this makes the text part as a separate element '''
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
#%% 3D scatter plot
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.scatter(df['Tcondensor'], df['Tevaporator'], df['COP'], c=df['COP'],cmap='viridis')
# ax.set_xlabel('Condensor temperature')
# ax.set_ylabel('Evaporator temperature')
# ax.set_zlabel('COP')

#%% Surface plot
from scipy.interpolate import griddata

xi = np.linspace(df['Tevaporator'].min(), df['Tevaporator'].max(), 100)
yi = np.linspace(df['Tcondensor'].min(), df['Tcondensor'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolating data
zi = griddata((df['Tevaporator'],df['Tcondensor']), df['COP'], (xi, yi), method='linear')

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # Surface plot
# surf = ax.plot_surface(xi, yi, zi, cmap='viridis', clim=(2.2,9),  edgecolor='none')
# ax.set_xlabel('Evaporator Temperature')
# ax.set_ylabel('Condensor Temperature')

# ax.set_zlabel('COP')
# fig.colorbar(surf)

#%% 2D surface plot. Remove levels i fyou dont mind a courser image
# zi = np.nan_to_num(zi, nan=np.nanmin(zi))
levels = np.linspace(np. nanmin(zi), np. nanmax(zi), 300)

plt.figure(figsize=(8, 6))
cp = plt.contourf(xi, yi, zi, levels=levels, cmap='viridis',vmin=2.1, vmax=9.2)  # Use contourf for filled contours
cbar = plt.colorbar(cp)  # Show a color bar

# cbar.set_ticks([2.1, 9.2])
plt.xlabel('Evaporator inlet temperature [degC]')
plt.ylabel('Condensor inlet temperature [degC]')
plt.title('Performance map of HP')

#%%
def plot_performance_map():
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(xi, yi, zi, levels=levels, cmap='viridis', vmin=2.1, vmax=9.2)
    cbar = plt.colorbar(cp, ax=ax)

    ax.set_xlabel('Evaporator inlet temperature [degC]')
    ax.set_ylabel('Condensor inlet temperature [degC]')
    ax.set_title('Performance map of HP')

    return fig, ax

fig,ax = plot_performance_map()
#%%
# fig, ax = plt.subplots()
# plt.scatter(df['Tevaporator'],df['Tcondensor'], edgecolors= "black", facecolors='none', linewidth=0.6)
