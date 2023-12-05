# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:37:03 2023

@author: 20181270
"""

import sys
import time                 # to measure the computation time
import os 
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #__file__: built-in ocnstant containing pathname of the current file
dir_main = os.getcwd()
dir_lib = dir_main + '\\src'
sys.path.append(dir_lib)
from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots
from PlotGroups import PlotGroups
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

directory = os.path.dirname(os.path.realpath(__file__))+'\\'
folder = 'res\\trn\\'
# directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\PhD\\TRNSYS\\Publication1\\'
# folder = 'Restart\\'
file = 'test20'
prefix = directory + folder + file

t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

#%%
if 'cp' in file:
    controls, energy, temp_flow = pf.cal_base_case(prefix)
    
else:
    temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter=",",index_col=0)
    energy = pd.read_csv(prefix+'_energy.txt', delimiter=",", index_col=0)
    controls = pd.read_csv(prefix+'_control_signal.txt', delimiter=",",index_col=0)
    
    controls = pf.modify_df(controls, t_start, t_end)
    temp_flow = pf.modify_df(temp_flow, t_start, t_end)
    energy = pf.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
    energy = pf.cal_energy(energy, controls)

occ = pd.read_csv(directory+folder+'occ.txt', delimiter=",",index_col=0)
occ = pf.modify_df(occ, t_start, t_end)
controls = pd.concat([controls,occ],axis=1)

temp_flow = pf.unmet_hours(controls, temp_flow)

energy_monthly, energy_annual = pf.cal_integrals(energy)

el_bill, gas_bill = pf.cal_costs(energy)
el_em, gas_em = pf.cal_emissions(energy)
pl,pe = pf.peak_load(energy)
rldc,ldc = pf.cal_ldc(energy)
opp_im, opp_ex, import_in, export_in = pf.cal_opp(rldc)
cop = pf.cal_cop(energy)
# penalty, energy = pf.cal_penalty(energy)

#%% plots
pt = Plots(controls, energy, temp_flow)
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2001,1,8, 0,0,0)
pt.plot_q(t1, t2)

#%% plotlt plot for residual LDC
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
fig = make_subplots(rows=1)

color_scale = plotly.colors.sequential.Viridis
color_scale = plotly.colors.qualitative.Bold

data = rldc['net_import']
fig.add_trace(go.Scatter(x=rldc.index, 
                          y=data, 
                          text=rldc['timestamp'],
                          hoverinfo="y+text"
                          )
               )
fig.show()

#%%
c1, e1, tf1= {}, {}, {}
r1,l1 = {}, {}
occ = pd.read_csv(directory+folder+'occ.txt', delimiter=",",index_col=0)
occ = pf.modify_df(occ, t_start, t_end)
#%% run multiple files
files = ['test4_cp','test7_cp','837_cp','test5_cp','test6_cp','test8_cp','test9_cp','test10_cp',
         '249', 'test11', 'test3','test12','test13','test2','test14','test15','test16'
         ,'test17','test18','test19','test20','test21','test22','396','test23','test24']

if 'e' in locals():                                     #for running the cell again
    existing_keys = list(e1.keys())                     #without having to run all labels again
    new_files = list(set(files)-set(existing_keys))     #check which labels exist and isolate the new ones
else:                                                   #if its a new run, then run all labels
    new_files = files

for file in new_files:
    print(file)
    prefix = directory + folder + file
    controls, energy, temp_flow = pf.create_dfs(file,prefix)
        
    controls = pd.concat([controls,occ],axis=1)
    temp_flow = pf.unmet_hours(controls, temp_flow)
    rldc,ldc = pf.cal_ldc(energy)
    
    c1[file] = controls
    e1[file] = energy
    tf1[file] = temp_flow
    r1[file] = rldc
    l1[file] = ldc
#%% create filtered dictionary
files = ['396','test23','test24']
c = {key: c1[key] for key in files}
e = {key: e1[key] for key in files}
tf = {key: tf1[key] for key in files}
#%%
t1 = datetime(2001,1,11, 0,0,0)
t2 = datetime(2001,1,12, 0,0,0)

fig, axs = plt.subplots(len(e), 1, figsize=(8, 2*len(e)))
axs0 = [ax.twinx() for ax in axs]
for i, label in enumerate(e):
    pt = Plots(c[label],e[label],tf[label])
    e[label]['Qhp'].plot(ax=axs[i])
    # e[label][['Qheat_living1','Qheat_living2']].sum(axis=1).plot(ax=axs[i], label='Qheat')
    c[label][['heatingctr1','heatingctr2']].plot(ax=axs[i],style='--', color=['lightskyblue','dodgerblue'])
    tf[label]['unmet'].plot(ax=axs[i],style='--', color='black')
    tf[label][['Tfloor1','Tfloor2','Thp_load_out']].plot(ax=axs0[i], 
                                          color=['mediumvioletred', 'darkmagenta','green'])
    temp_flow[['Tset1','Tset2']].plot(ax=axs0[i], style='--', 
                                              color=['mediumvioletred', 'darkmagenta'])
    pf.plot_specs(axs[i], t1,t2, 0,5, 'energy[kWh]',legend_loc='upper left')
    pf.plot_specs(axs0[i], t1,t2, -15,34, 'Temperature [deg C]',legend_loc='upper right')
    heating_demand = round(e[label]['Qheat'].sum()*0.1, 2)
    unmet = round(tf[label]['unmet'].sum(),2)
    axs[i].set_title(label)
    print(label + '(heat) ='+ str(heating_demand) + ' kWh, unmet hours = '+str(unmet)+' Qhp = '+str(e[label]['Qhp'].sum()*0.1))

#%% cp - plots
t1 = datetime(2001,1,10, 0,0,0)
t2 = datetime(2001,1,12, 0,0,0)

pt = Plots(controls, energy, temp_flow)
fig,(ax, ax2) = plt.subplots(2,1)
ax0 = ax.twinx()
# energy[['Qheat_living1','Qheat_living2']].sum(axis=1).plot(ax=ax)
energy[['Qhp','gas']].plot(ax=ax)
temp_flow[['Tfloor1','Tfloor2']].plot(ax=ax0,color=['mediumvioletred', 'darkmagenta'])
temp_flow[['Tset1','Tset2']].plot(ax=ax0, style='--', color=['mediumvioletred', 'darkmagenta'])
pf.plot_specs(ax, t1,t2, 0,7, 'energy[kWh]',legend_loc='upper left')
pf.plot_specs(ax0, t1,t2, -10,30, 'Temperature [deg C]',legend_loc='upper right')



#%%
t1 = datetime(2001,1,10, 0,0,0)
t2 = datetime(2001,1,12, 0,0,0)
fig,(ax, ax2)= plt.subplots(2,1)
ax0 = ax.twinx()
temp_flow[['Tfloor1','Tfloor1_2']].plot(ax=ax)
temp_flow['Tset1'].plot(ax=ax, style=':', color='C0')
energy['Qrad1'].plot(ax=ax0, style ='--', color='black')
pf.plot_specs(ax, t1,t2, 12,23, 'Temperature [deg C]',legend_loc='upper left', 
              title='Is heating effective in the two zones of the living room')
pf.plot_specs(ax0, t1,t2, -0.1,20, 'Energy delivered by Radiator [kW]',legend_loc='upper right')
ax.grid(which='both',linestyle='--', alpha=0.4)

ax20 = ax2.twinx()
temp_flow[['Thp_load_out','Trad1_in','Trad1_return']].plot(ax=ax2)
energy[['Qhp_heating_out','Qrad1']].plot(ax=ax20, style='--')
pf.plot_specs(ax2, t1,t2, 40,70, ylabel='t',legend_loc='upper left')
pf.plot_specs(ax20, t1,t2, -0.1,25, 'Energy delivered by Radiator [kW]',legend_loc='upper right')

#%% namual calculation of Qrad
q_rad = temp_flow['mrad1_in']*4.182*(temp_flow['Trad1_in']-temp_flow['Trad1_return'])/3600
q_hp = temp_flow['mhp_load_out']*4.182*(temp_flow['Thp_load_out']-temp_flow['Thp_load_in'])/3600
fig,ax = plt.subplots()
q_hp.plot(ax=ax)
energy['Qhp_heating_out'].plot(ax=ax)#,style='--',linewidth=0.3)
diff = (energy['Qhp_heating_out']-q_rad)

t1 = datetime(2001,1,10, 0,0,0)
t2 = datetime(2001,1,17, 0,0,0)
fig2,ax2 = plt.subplots()
temp_flow['mhp_load_out'].plot(ax=ax2)
temp_flow['mhp_load_in'].plot(ax=ax2, style='--')
ax2.set_xlim([t1,t2])