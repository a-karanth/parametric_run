# -*- coding: utf-8 -*-
"""
Plotting the performance map of heat pump

@author: 20181270
"""
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def read_data(folder, cdata, hdata): # read the raw data from the txt files
    with open(hdata, 'r') as file:
        lines = file.readlines()
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
    return data


def find_input_parameters(data):
    inputs = {}
    for i in range(4):
        key = data[i][-1]
        values = data[i][:-1]
        inputs[key] = values
    return inputs

def make_df_order_of_imput_parameters(inputs):
    """
    For now you have to manually define what the order is.
    From observation, order of keys remains the same.
    input_rows.append line changes based on the format of the t
    """

    order_of_keys = ['!Entering Source Temperatures',
                     '!Entering Load Temperatures',
                     '!Normalized Source Flow Rate',
                     '!Normalized Load Flow Rate']
    input_rows = []
    for load_flow in inputs['!Normalized Load Flow Rates']:
        for source_flow in inputs['!Normalized Source Flow Rates']:
            for t_load in inputs['!Entering Load Temperatures (C)']:
                for t_source in inputs['!Entering Source Temperatures (C)']:
                    input_rows.append([load_flow, source_flow, t_load, t_source])
    return input_rows

def make_df_raw_data(data, input_rows):
    power_data = data[4:] 
    power_data = [[j for j in i if not isinstance(j, str)] for i in power_data]
    df_data = [row1 + row2 for row1, row2 in zip(input_rows, power_data)]
    
    columns = ['Fcond', 'Feva','Tcond_in','Teva','Capacity','Power']
    df = pd.DataFrame(data=df_data, columns=columns)   
    return df
    
def make_performance_df(df, rated_capacity, rated_power):
    # Input file has capacity, power and flow as normalized values
    # Scaling them to my inputs used in TRNSYS
    # Fcond_final, for calculation for Tcond_out, uses actual Fcond value
    cap_scale = rated_capacity/3600 #in kW
    pow_scale = rated_power/3600 # in kW
    load_flow_rate = 1400/3600 # in kg/s 
    df['Capacity'] = df['Capacity'] * cap_scale
    df['Power'] = df['Power'] * pow_scale
    
    df['COP'] = df['Capacity']/df['Power']
    df['Fcond_final'] = load_flow_rate
    df['Tcond_out'] = df['Capacity']*1000/(df['Fcond_final']* 4186) +  df['Tcond_in']
    return df

def cal_gridpoints_for_contourmap(df):
    df = df[(df['Feva']==0.9) & (df['Fcond']==0.9)]
    xi = np.linspace(df['Teva'].min(), df['Teva'].max(), 100)
    yi = np.linspace(df['Tcond_in'].min(), df['Tcond_in'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolating data
    zi = griddata((df['Teva'],df['Tcond_in']), df['COP'], (xi, yi), method='linear')
    return xi, yi,zi

def plot_performance_map(xi,yi,zi, dim='2D'):
    if dim == '2D':
        levels = np.linspace(np. nanmin(zi), np. nanmax(zi), 50)
        fig, ax = plt.subplots(figsize=(8, 6))
        # cp = plt.contourf(xi, yi, zi, levels=levels, cmap='turbo',vmin=2.98, vmax=8.3, alpha=0.7)  # Use contourf for filled contours
        cp = plt.contourf(xi, yi, zi, levels=levels, cmap='turbo',vmin=1.78, vmax=5, alpha=0.7)  # Use contourf for filled contours
        cbar = plt.colorbar(cp, ax=ax)  # Show a color bar
        # cbar.set_ticks([2.1, 9.2])
        ax.set_xlabel('Evaporator inlet temperature [degC]')
        ax.set_ylabel('Condensor outlet temperature [degC]')
        ax.set_title('Performance map of HP')
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # Surface plot
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', clim=(2.2,9),  edgecolor='none')
        ax.set_xlabel('Evaporator Temperature')
        ax.set_ylabel('Condensor Temperature')
        ax.set_zlabel('COP')
        fig.colorbar(surf)
    return fig,ax

def plot_hp_performance():    
    folder = 'C:\\TRNSYS18\\Tess Models\\SampleCatalogData\\Water-to-WaterHeatPumps\\Normalized\\'
    cdata = folder + 'WWHP_Cooling-Normalized.dat'
    hdata = folder + 'WWHP_Heating-Normalized_Ecoforest.dat'
    data = read_data(folder, cdata, hdata)
    inputs = find_input_parameters(data)
    input_rows = make_df_order_of_imput_parameters(inputs)
    df = make_df_raw_data(data, input_rows)
    df = make_performance_df(df, 21600, 7200)
    xi, yi, zi = cal_gridpoints_for_contourmap(df)
    fig,ax = plot_performance_map(xi,yi,zi)
    return fig,ax, df