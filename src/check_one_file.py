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
from datetime import datetime, timedelta
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

directory = os.path.dirname(os.path.realpath(__file__))+'\\'
folder = 'res\\trn\\'
# directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\PhD\\TRNSYS\\Publication1\\'
# folder = 'Restart\\'
label = 'x'
file = directory + folder + label

inputs = pd.read_csv(folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
t_start = datetime(2001,2,9, 0,0,0)
t_end = datetime(2001,2,17, 0,0,0)

#%% Read files
controls, energy, temp_flow = pf.create_dfs(label,file)

mass_bal, param = pf.create_additional_dfs(file)
occ = pd.read_csv(directory+folder+'occ.txt', delimiter=",",index_col=0)
occ = pf.modify_df(occ)
occ = occ[:energy.index[-1]] #make occ dataframe as long as the rest of the dfs. especially for incomplete sims
controls = pd.concat([controls,occ],axis=1)

temp_flow = pf.unmet_hours(controls, temp_flow)

energy_monthly, energy_annual = pf.cal_integrals(energy)

el_bill, gas_bill = pf.cal_costs(energy)
el_em, gas_em, energy = pf.cal_emissions(energy)
pl,pe = pf.peak_load(energy)
rldc,ldc = pf.cal_ldc(energy)
opp_im, opp_ex, import_in, export_in = pf.cal_opp(rldc)
cop = pf.cal_cop(energy)
penalty, energy = pf.cal_penalty(energy)

#%% plots
import matplotlib.ticker as ticker

pt = Plots(controls, energy, temp_flow)
t1 = datetime(2001,1,7, 0,0,0)
t2 = datetime(2001,1,8, 0,0,0)

fig = pt.check_sim(t1,t2,file,ssbuff=False)

test = pd.DataFrame()
test['Thp_source'] = temp_flow['Thp_source_in']*controls['coll_pump']*controls['hx_bypass']
test['Thx_source'] = temp_flow['Thp_source_in']*controls['coll_pump']*(controls['hx_bypass']==0)

figx = pt.plot_monthly(file)
print(mass_bal.sum())

figy, (axy, axz) = plt.subplots(2,1)
axy0 = axy.twinx()
temp_flow['mssbuff_load'].plot.area(ax=axy0, alpha=0.2, color='darkred')
temp_flow['mssbuff_source'].plot.area(ax=axy0, alpha=0.2, color='lightseagreen')
temp_flow[['Tssbuff_load_out','Tssbuff_load_in','Tssbuff_source_out','Tssbuff_source_in','Tavg_ssbuff']].plot(ax=axy, color=['darkred','indianred','lightseagreen','teal','black'])
# temp_flow['Tcoll_in'].plot(ax=axy, color='gray', linewidth=2, style='--')
controls[['hx_bypass','ssbuff_stat']].plot(ax=axy, color=['black','red'], style='--')
axy.legend(loc='upper left')
axy.set_ylabel('Temperature [degC]')
axy0.set_ylabel('mass flow [kg/hr]')
axy0.legend(loc='upper right')
axy0.set_ylim([0,10000])
#test of setting ticks only upto 800
def custom_ticks(value, pos):
    if value > 800:
        return ''
    return int(value)

axy0.yaxis.set_major_formatter(ticker.FuncFormatter(custom_ticks))
axy0.yaxis.set_major_locator(ticker.MultipleLocator(800))  # Add ticks up to 800

fig.autofmt_xdate()
axy.grid(linestyle='--', alpha=0.5)

axz0 = axz.twinx()
energy[['Qssbuff_source','Qssbuff_load']].plot(ax=axz, color=['lightseagreen', 'darkred'])
mass_bal[['coll_pump','hp_source_pump']].plot.area(ax=axz0,color=['lightseagreen', 'darkred'], alpha=0.2, stacked=False)
axz.legend(loc='upper left')
axz.grid(linestyle='--', alpha=0.5)
axz0.set_ylabel('mass balance error')
axz.set_ylabel('Energy tranfser [kW]')
axz0.legend(loc='upper right')

for ax in figy.axes:
    ax.set_xlim([t1,t2])
# Redraw the figure to update the display
figy.canvas.draw()

#%% checking the unique control states
# make controls specialized dataframe
spec = controls[['dhw_demand', 'ctr_dhw','ctr_sh_buff', 'ssbuff_stat', 'ctr_irr','ctr_coll_t','coll_pump', 'ctr_hp','hx_bypass', 'load_hx_bypass','demand','div_load','tset_dhw','coll_pump2']]
spec['operation_window'] = ((temp_flow['Tssbuff_load_out'] > -25) & (temp_flow['Tssbuff_load_out'] < 35))*1
unique = spec.drop_duplicates()
left = unique[['dhw_demand', 'ctr_dhw','ctr_sh_buff', 'operation_window','ssbuff_stat', 'ctr_irr','ctr_coll_t','coll_pump','tset_dhw','coll_pump2']]
unique_left = left.drop_duplicates()
right = unique[['ctr_hp','hx_bypass', 'load_hx_bypass','demand','div_load','tset_dhw']]
right_unique = right.drop_duplicates()
new_spec = controls[['coll_pump','ctr_irr', 'demand','ssbuff_stat', 'op_window','ctr_hp', 'hx_bypass','tset_dhw','coll_pump2','ctr_dhw']].drop_duplicates()
#%% plotting percetnatge of occurance of different modes - mostly based on right hand values
controls['operation_mode'] = None

conditions = [
    (controls['coll_pump'] == 0)  & (controls['ctr_hp'] == 0) & (controls['demand'] == 0),
    (controls['coll_pump'] == 1) & (controls['hx_bypass'] == 1) & (controls['ctr_hp'] == 0) & (controls['demand'] == 0),
    (controls['coll_pump'] == 0) & (controls['ctr_hp'] == 1) & (controls['demand'] == 1),
    (controls['coll_pump'] == 1) & (controls['hx_bypass'] == 1) & (controls['ctr_hp'] == 1) & (controls['demand'] == 1),
    (controls['coll_pump'] == 1) & (controls['hx_bypass'] == 1) & (controls['ctr_hp'] == 0) & (controls['demand'] == 1),
    (controls['coll_pump'] == 0) & (controls['ctr_hp'] == 0) & (controls['demand'] == 1),
    (controls['coll_pump'] == 1) & (controls['hx_bypass'] == 0) & (controls['ctr_hp'] == 0) & (controls['demand'] == 1),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 0),
    (controls['coll_pump'] == 1) & (controls['hx_bypass'] == 0) & (controls['ctr_hp'] == 0) & (controls['demand'] == 0) #load side is off, so energy is wasted/curtailed
]

# Define corresponding operation_mode values
values = [0, 1, 2, 3, 4, 5, 6, 7,0]
# Use np.select to assign values based on conditions
controls['operation_mode'] = np.select(conditions, values, default=None)

percentage_occurrence = controls['operation_mode'].value_counts(normalize=True) * 100

# Sort by index to ensure the order is [0, 1, 2, 3, 4, 5, 6, 7]
percentage_occurrence = percentage_occurrence.sort_index().drop(index=0)
percentage_df = percentage_occurrence.to_frame().T
percentage_df.index = [None]
n_col = len(percentage_df.columns)

from matplotlib.cm import get_cmap
fig, ax = plt.subplots(figsize=(10, 3))
cmap = get_cmap('PiYG', n_col)
colors = [cmap(i) for i in range(n_col)]
percentage_df.plot.barh(ax=ax, stacked=True, color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Percentage Occurrence')
ax.set_title('Percentage Occurrence of Each Operation Mode')
ax.legend(title='Operation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%% environmental conditions driven operation models
controls['operation_mode2'] = None
conditions = [
    (controls['coll_pump'] == 0) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) ,
    (controls['coll_pump'] == 0) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 1),
    (controls['coll_pump'] == 0) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 0) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 0) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 1),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 0) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 1),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 0) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 1),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 0) & (controls['tset_dhw'] == 70),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 1) & (controls['tset_dhw'] == 70),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 0) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 1) & (controls['tset_dhw'] == 70),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 0) & (controls['ctr_hp'] == 0),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['ssbuff_stat'] == 1) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 1),
    (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) & (controls['ssbuff_stat'] == 0) & (controls['op_window'] == 1) & (controls['ctr_hp'] == 0)
]

values = [0, 1, 2, 1, 3, 3, 4, 5, 6, 7, 7, 3, 3, 7, 5, 6, 8]
values = [0, 1, 2, 1, 3, 3, 8, 7, 6, 4, 4, 3, 3, 4, 7, 6, 5]
controls['operation_mode2'] = np.select(conditions, values, default=None)

percentage_occurrence = controls['operation_mode2'].value_counts(normalize=True) * 100
percentage_occurrence = percentage_occurrence.sort_index().drop(index=0)
percentage_df = percentage_occurrence.to_frame().T
percentage_df.index = [None]
n_col = len(percentage_df.columns)

#%%
from matplotlib.cm import get_cmap
fig, ax = plt.subplots(figsize=(10, 3))
cmap = get_cmap('PiYG', n_col)
colors = [cmap(i) for i in range(n_col)]
percentage_df.plot.bar(ax=ax, stacked=True, color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Percentage Occurrence')
ax.set_title('Percentage Occurrence of Each Operation Mode')
ax.legend(title='Operation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%% calculated operation modes - for with ss_buff and without ss_buff
controls['operation_mode'] = None
main_op_mode = p[key]['operation_mode']
if main_op_mode==4:
    conditions = [
        (controls['coll_pump'] == 1)  & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 0), 
        (controls['coll_pump'] == 1)  & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 1), 
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['hx_bypass'] == 0),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & ((controls['demand'] == 0) or controls['tset_dhw']==70),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 0),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 0) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) ,
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 0),
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 0) & (controls['demand'] == 0) 
    ]
    
    # Define corresponding operation_mode values
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8]
    
else:
    conditions = [
        (controls['coll_pump'] == 1)  & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['op_window'] == 1) & (energy['Qaux_hp'] == 0), 
        (controls['coll_pump'] == 1)  & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['op_window'] == 1) & (energy['Qaux_hp'] == 1), 
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['demand'] == 1) & (controls['op_window'] == 0), #buffer charges
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 1) & (controls['ss_stat'] == 0) & ((controls['demand'] == 0) or controls['tset_dhw']==70),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 0),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 0) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 1) & (controls['ctr_irr'] == 0) & (controls['demand'] == 0) ,
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 0),
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (controls['hx_bypass'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 0) & (controls['demand'] == 1) & (energy['Qaux_hp'] == 1),
        (controls['coll_pump'] == 0) & (controls['demand'] == 0) 
    ]
    
    # Define corresponding operation_mode values
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8]
# Use np.select to assign values based on conditions
controls['operation_mode'] = np.select(conditions, values, default=None)

percentage_occurrence = controls['operation_mode'].value_counts(normalize=True) * 100

# Sort by index to ensure the order is [0, 1, 2, 3, 4, 5, 6, 7]
percentage_occurrence = percentage_occurrence.sort_index().drop(index=0)
percentage_df = percentage_occurrence.to_frame().T
percentage_df.index = [None]
n_col = len(percentage_df.columns)

from matplotlib.cm import get_cmap
fig, ax = plt.subplots(figsize=(10, 3))
cmap = get_cmap('PiYG', n_col)
colors = [cmap(i) for i in range(n_col)]
percentage_df.plot.barh(ax=ax, stacked=True, color=colors, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Percentage Occurrence')
ax.set_title('Percentage Occurrence of Each Operation Mode')
ax.legend(title='Operation Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()                
#%% Change xlim of all axes
t1 = datetime(2001,6,13, 0,0,0)
t2 = datetime(2001,6,14, 0,0,0)
for ax in fig_cum.axes:
    ax.set_xlim([t1,t2])
# Redraw the figure to update the display
figy.canvas.draw()
 
# test for mass balace in collector

fig, csh, f, ret = pt.plot_coll_loop_ss_buff(t1,t2)
# coll_pump test, collector coll_in vs coll_out
plt.figure()
temp_flow['mcoll_in'].plot.area(alpha=0.2)
temp_flow['mcoll_out'].plot(style='--', linewidth=2)
plt.legend()
plt.xlim([t1,t2])
mb['coll_pump'].plot(ax=csh, marker='^', color='r', linewidth=0.5)

fig_coll, c = plt.subplots()
temp_flow[['Tamb','Tcoll_in']].plot(ax=c)
c0 = c.twinx()
c_irr = controls['ctr_irr']/3
c_coll_t = controls['ctr_coll_t']*(2/3)
controls['coll_pump'].plot(ax=c0, color='gray',style='--')
c_irr.plot(ax=c0, color='orange', style='--')
c_coll_t.plot(ax=c0, color='red', style='--')
energy['QuColl'].plot.area(ax=c0, color='gold', alpha=0.2, stacked=False)
c.set_xlim([t1,t2])

#%% radiator and collector test test
df = pd.DataFrame()
df['Trad1'] = temp_flow['Trad1_in']*(temp_flow['mrad1_in']>0)
df['Trad2'] = temp_flow['Trad2_in']*(temp_flow['mrad2_in']>0)
df = df.replace(0,np.NaN)
figr, (rad,coll) = plt.subplots(2,1)
df[['Trad1','Trad2']].plot(ax=rad, color=['firebrick', 'teal'])
temp_flow[['Tfloor1','Tfloor2']].plot(ax=rad, color=['firebrick', 'teal'], style=':')


# coll0 = coll.twinx()
# controls['hx_bypass'].plot(ax=coll0, color='black', style=':')
# controls['ssbuff_stat'].plot(ax=coll0, color='orange', style='--')
temp_flow[['Tssbuff_source_out','Tcoll_in']].plot(ax=coll)
temp_flow['Thx_source_out'].plot(ax=coll, style='--')
coll.legend(loc='upper left')         
coll.grid(axis='y',which='both',alpha=0.2,color='black')

t1 = datetime(2001,1,5, 12,0,0)
t2 = datetime(2001,1,5, 18,0,0)
for ax in fig1.axes:
    ax.set_xlim([t1,t2])
# # Redraw the figure to update the display
# figr.canvas.draw()
#%% initialize for multiple files
c1, e1, tf1= {}, {}, {}
r1,l1 = {}, {}
m,mb,p = {},{}, {}
occ = pd.read_csv(directory+folder+'occ.txt', delimiter=",",index_col=0)
occ = pf.modify_df(occ)
#%% run multiple files
files = [#'2000',
          'test70','test71',
          'test75','test76', 'test77']
        # '2008', '2009', '2010', '2011', '2012', '2013']

if 'e' in locals():                                     #for running the cell again
    existing_keys = list(e1.keys())                     #without having to run all labels again
    new_files = list(set(files)-set(existing_keys))     #check which labels exist and isolate the new ones
else:                                                   #if its a new run, then run all labels
    new_files = files

for label in new_files:
    print(label)
    file = directory + folder + label
    controls, energy, temp_flow = pf.create_dfs(label,file)
    mass_balance,param = pf.create_additional_dfs(file)
    energy_monthly, energy_annual = pf.cal_integrals(energy)
    controls = pd.concat([controls,occ],axis=1)
    energy = pf.cal_cost_df(energy)
    el_em,gas_em, energy = pf.cal_emissions(energy)
    # rldc,ldc = pf.cal_ldc(energy)
    energy_monthly.index = energy_monthly.index.strftime('%b')
    
    c1[label] = controls
    e1[label] = energy
    tf1[label] = temp_flow
    # r1[label] = rldc
    # l1[label] = ldc
    m[label] = energy_monthly
    mb[label] = mass_balance
    p[label] = param

#%% create filtered dictionary
# files = ['2008', '2009', '2010', '2011', '2012', '2013']
c = {key: c1[key] for key in files}
e = {key: e1[key] for key in files}
tf = {key: tf1[key] for key in files}
pt = {}

#%% plot check sims
key = 'test76'
if 'operation_mode' not in p[key].columns or p[key]['operation_mode'].iloc[0]!=4:
    ssbuff=False
else:
    ssbuff=True
pt = Plots(c[key], e[key], tf[key])
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2001,1,9, 0,0,0)
# qtot = round(m[key]['Qheat'].sum(),0)
fig1 = pt.check_sim(t1,t2,key,ssbuff)
fig2 = pt.plot_hx_hp_loops(t1,t2)
# fig_flow, ax_flow = plt.subplots()
# ax_flow0 = ax_flow.twinx()
# tf[key][['Tcoll_in','Tcoll_out','Tamb']].plot(ax=ax_flow0)
# e[key]['Qirr'].plot(ax=ax_flow0, color='gold', linewidth=2)
# tf[key][['mcoll_in','mcoll_out']].plot(ax=ax_flow, style='--')
# tf[key]['mcoll_in'].plot(ax=ax_flow,style='--')
# ax_flow0.set_xlim([t1,t2])
# ax_flow0.legend(loc='upper left')
# ax_flow.legend(loc='center left')

# figx = pt.plot_monthly(key)
# t1 = datetime(2001,1,1, 0,0,0)
# t2 = datetime(2001,1,7, 12,0,0)
fig_hp,ax_hp = pt.plot_cop_on_map(key)
fig_hp,ax_hp = pt.plot_hp_operation_on_map(key)

# check daily totals
# pf.check_daily_totals(t1, t2, e[key], ['QuColl_irr','QuColl_t'])
# fig2 = pt.plot_coll_balance(mb[key], t1, t2)
# fig2.suptitle(key)

#%%
t1 = datetime(2001,6,14, 0,0,0)
t2 = datetime(2001,6,15, 0,0,0)
for ax in fig1.axes:
    ax.set_xlim([t1,t2])
#%% Qhp + Qaux_hp is greater than Qdhw_source + Qsh_source
key='test68'
energy = e[key]
instances = energy[(energy['Qhp_load'] + energy['Qaux_hp']) > (energy['Qdhw_source'] + energy['Qsh_buff_source'])]
in_heat = energy['Qhp_load']+energy['Qaux_hp']
out_heat = -(energy['Qdhw_source'] + energy['Qsh_buff_source'])
qhx = energy['Qhx']
qu = energy['QuColl']
fig,ax = plt.subplots()
ax.plot(in_heat.index, in_heat,linestyle='--', label = 'Qhp_out + Qaux_hp')
ax.plot(qhx.index, qhx,linestyle='--', color='orange', label='Qhx')
ax.plot(qu.index, qu,linestyle='--', color='red',label='QuColl')
ax.fill_between(out_heat.index, out_heat, alpha=0.2, label='Qdhw_in + Qsh_in')
t1 = datetime(2001,6,13, 0,0,0)
t2 = datetime(2001,6,14, 0,0,0)
ax.set_xlim([t1,t2])
ax.legend()
#%% mcoll_plot in plotly
import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = 'browser'
import plotly.express as px
from plotly.subplots import make_subplots

# Create the figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add the first set of lines to the secondary y-axis
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_in'], mode='lines', name='Tcoll_in'), secondary_y=True)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_out'], mode='lines', name='Tcoll_out'), secondary_y=True)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tamb'], mode='lines', name='Tamb'), secondary_y=True)

# Add the second set of lines to the primary y-axis with dashed style
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['mcoll_in'], mode='lines', name='mcoll_in', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['mcoll_out'], mode='lines', name='mcoll_out', line=dict(dash='dash')))

# Update layout to set the range and axes
fig.update_layout(
    xaxis=dict(range=[t1, t2]),
    yaxis2=dict(title='Temperature', overlaying='y', side='right', range=[-2,5]),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

# Show the plot in the browser
fig.show()


#%% monthly plot comparison
compare1 = 'test71'
compare2 = 'test76'
pf.compare_monthly_bars(m, compare1,compare2)

#%% compare T HP source in of multiple cases
compare = ['test64', 'test69']
for key in compare:
    tf[key]['Thp_source_in_final'] = c[key]['ctr_hp'] * tf[key]['Thp_source_in']
figc, axs = plt.subplots(len(compare), 1, figsize=(10, 3 * len(compare)))  # Adjust figsize as necessary
df = tf
column='Thp_source_in_final'

if len(compare) == 1:
    axs = [axs]  # Makes axs iterable if only one subplot is created

for ax, comp in zip(axs, compare):
    pt = Plots(c[comp], e[comp], tf[comp])
    # pt.plot_colormap(ax, df[comp][column], comp, cmap='PiYG_r', vmin=1.8, vmax=2.8)
    pt.plot_colormap(ax, df[comp][column], comp, cmap='PuOr_r', vmin=-25, vmax=35)
    # if COP calculate the max value that occurs in 99% of the data by df[comp][column].quantile(0.99)

figc.suptitle(column)
plt.tight_layout()
plt.show()

#%% compare annual cop of multiple cases
compare = ['test64', 'test69']
figc, axs = plt.subplots(len(compare), 1, figsize=(10, 3 * len(compare)))  # Adjust figsize as necessary

if len(compare) == 1:
    axs = [axs]  # Makes axs iterable if only one subplot is created

for ax, comp in zip(axs, compare):
    pt = Plots(c[comp], e[comp], tf[comp])
    pt.plot_colormap(ax, e[comp]['COP'], comp, cmap='viridis_r', vmin=2.1, vmax=2.9)
    Q1 = round(e[comp]['COP'].quantile(0.25),2)
    median = round(e[comp]['COP'].median(),2)
    Q3 = round(e[comp]['COP'].quantile(0.75),2)
    print(f'{comp}, Q1={Q1}, median={median}, Q3={Q3}')

plt.tight_layout()
plt.show()

#%% plot new controller for inlet of collecctor
compare = ['test49', 'test50']
compare1 = 'test49'
compare2 = 'test50'

fig_ctr,ax_ctr = plt.subplots()
c[compare1]['coll_in_thresh'].plot(ax=ax_ctr, linewidth=2)
c[compare2]['coll_in_thresh'].plot.area(ax=ax_ctr, alpha=0.2)
ax_ctr.legend()
t1 = datetime(2001,1,23, 0,0,0)
t2 = datetime(2001,1,31, 0,0,0)
ax_ctr.set_xlim([t1,t2])

#%% compare daily/monthly cumulative results of 2 or more cases
compare = ['test64','test69']
which = 'monthly'
pf.plot_energy_cumulatives(compare, e, which)

pf.plot_kpi_cumulatives(compare, e, which)

#%% collector loss coeff
t1 = datetime(2001,6,6, 0,0,0)
t2 = datetime(2001,6,7, 0,0,0)
df = tf['test62'][['coll_loss_coeff','apparent_coll_loss_coeff']].copy()
df = df/3.6 # converting from kJ/hr to J/s(W)
fig_loss, ax_loss = plt.subplots()
df['coll_loss_coeff'].plot(ax=ax_loss, color='black', style='--')
df['apparent_coll_loss_coeff'].plot(ax=ax_loss, color='red', style=':')
ax_loss.set_xlim([t1,t2])
ax_loss.grid(linestyle='--', alpha=0.4)
ax_loss.legend()
ax_loss.set_ylabel('Thermal loss coefficient [W/m2K]')

#%% plot energy balance for load side of HP and manipulate Qss_buff_source for plotting
file = 'test67'
energy = e[file]
temp_flow = tf[file]
# energy['Qsh_buff_source'][energy['Qsh_buff_source']<0] = 0
# energy['Qdhw_source'][energy['Qdhw_source']<0] = 0
energy['Qsh_buff_source'] = energy['Qsh_buff_source']*(-1)
energy['Qdhw_source'] =  energy['Qdhw_source']*(-1)
fig_bal, ax_bal = plt.subplots()
energy[['Qsh_buff_source','Qdhw_source']].plot.area(ax=ax_bal, alpha=0.5, stacked=False, color=['tab:blue','tab:orange'] )
energy[['Qhp_load','Qaux_hp','Qhx']].sum(axis=1).plot(ax=ax_bal, style='--', color='black')
controls['div_load'].plot(ax=ax_bal,color='green', style=":")
controls['demand'].plot.area(ax=ax_bal,color='green', alpha=0.2)

# ax_bal0 = ax_bal.twinx()
# temp_flow[['Tsh_source_in','Tsh_source_out']].plot(ax=ax_bal0,color=['mediumorchid', 'indigo'])
# temp_flow[['Tdhw_source_in','Tdhw_source_out']].plot(ax=ax_bal0,color=['chocolate', 'saddlebrown'])
# temp_flow[['Thx_load_out','Thx_load_in']].plot(ax=ax_bal0,color=['lightgreen', 'darkgreen'])
ax_bal.set_xlim([t1,t2])
ax_bal.set_ylim([-14,14])
# ax_bal0.legend(loc='upper right')
ax_bal.legend(loc='upper left')
fig_bal.suptitle(file)

#%% energy balance of a tank - dhw
file = 'test65'
energy = e[file]
fig_eb, ax_eb = plt.subplots()
energy[['Qdhw_source','Qdhw_load','Qaux_dhw','Qloss_dhw']].plot(ax=ax_eb)
energy['Qstored_dhw'].plot.area(ax=ax_eb, alpha=0.2, stacked=False)
ax_eb.set_xlim([t1,t2])
fig_eb.suptitle(file)
ax_eb.legend()

#%% hx energy vs temp
file = 'test67'
t1 = datetime(2001,3,18, 0,0,0)
t2 = datetime(2001,3,19, 0,0,0)
fig_hx, ax_hx = plt.subplots()
ax_hx0 = ax_hx.twinx()
e[file]['Qhx'].plot.area(ax=ax_hx,alpha=0.2,stacked=False, color='orange')
tf[file][['Thx_source_in','Thx_source_out','Thx_load_out','Thx_load_in']].plot(ax=ax_hx0, color=['black','gray','blue','lightblue'])
ax_hx.set_xlim([t1,t2])
ax_hx.set_title(file)

#%% check if QuColl is negative only when HX_bypass=0
file = 'test67'
t1 = datetime(2001,3,19, 0,0,0)
t2 = datetime(2001,3,20, 0,0,0)
energy = e[file]
controls = c[file]
negative_QuColl = energy[energy['QuColl'] < -0.02]

# Check if all corresponding hx_bypass values in the controls DataFrame are 0
all_negative_when_hx_bypass_zero = (controls.loc[negative_QuColl.index, 'hx_bypass'] == 0).all() # if false at least 1 exists that does not satisfy
mismatch_indices = negative_QuColl.index[controls.loc[negative_QuColl.index, 'hx_bypass'] != 0]
match_indices = negative_QuColl.index[controls.loc[negative_QuColl.index, 'hx_bypass'] == 0]

# Extract the relevant rows from both DataFrames
mismatch_rows = energy.loc[mismatch_indices]
mismatch_controls = controls.loc[mismatch_indices]

match_rows = energy.loc[match_indices]
match_controls = controls.loc[match_indices]

hx_positive = energy['Qhx'][energy['Qhx'] > 0].sum()
hx_negative = energy['Qhx'][energy['Qhx'] < 0].sum()
qu_negative = energy['QuColl'][energy['QuColl'] < 0].sum()
qu_positive = energy['QuColl'][energy['QuColl'] > 0].sum()

#%%
negative_qhx = energy['Qhx'][energy['Qhx'] < 0]
negative_qucoll = energy['QuColl'][energy['QuColl'] < 0]

# Align the indices of both series
aligned_qhx, aligned_qucoll = negative_qhx.align(negative_qucoll, join='inner')

# Find where the values are different
difference_mask = aligned_qhx != aligned_qucoll

# Extract and display the differences
differences = pd.DataFrame({
    'Qhx': aligned_qhx[difference_mask],
    'QuColl': aligned_qucoll[difference_mask]})

qucoll_negative_qhx_not_negative_mask = negative_qucoll.index.difference(negative_qhx.index)
# Extract and add the additional column
qucoll_negative_qhx_not_negative = negative_qucoll.loc[qucoll_negative_qhx_not_negative_mask]

differences['QuColl_Neg_Qhx_Not_Neg'] = None  # Initialize the column
# Update the differences DataFrame with the new information
differences = pd.concat([differences, qucoll_negative_qhx_not_negative.rename('QuColl_Neg_Qhx_Not_Neg')], axis=1)

positive_qucoll_irr = energy['QuColl_irr'][energy['QuColl_irr'] > 0].sum()
positive_qucoll_t = energy['QuColl_t'][energy['QuColl_t'] > 0].sum()

fig,ax = plt.subplots()
differences['QuColl'].plot(ax=ax)    
differences['QuColl_Neg_Qhx_Not_Neg'].plot(ax=ax,color='black')
negative_qucoll.plot.area(ax=ax,alpha=0.2)
#%% Sankey diagram
import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = 'browser'
file = 'test76'
day = datetime(2001,4,19,0,0,0)
en = pf.comp_en_for_one_day(e[file], day)

flows, unique_labels, label_to_index = pf.make_sankey_flows(en, p[file])

# Create the sources, targets, and values lists
sources = [flows[key]['source'] for key in flows]
targets = [flows[key]['target'] for key in flows]
values = [flows[key]['value'] for key in flows]

node_colors, link_colors = pf.sankey_node_colors(unique_labels, sources)
node_x, node_y = pf.sankey_node_positions(unique_labels)

node_totals, node_totals_load = pf.sankey_node_totals(unique_labels,sources,targets,values)

# Create the Sankey diagram
fig = go.Figure(go.Sankey(node=dict(pad=100,
                                    thickness=20,
                                    line=dict(color="black", width=0.5),
                                    label=unique_labels,
                                    color=node_colors,                                    
                                    x = node_x,
                                    y=node_y),  # Set y positions),
                          link=dict(source=sources,
                                    target=targets,
                                    value=values,
                                    color=link_colors)
                          ))

# Update the layout for better visualization
fig.update_layout(title_text=f"Energy Flow {file}",
                  font=dict(size=16),
                  hovermode='x',
                  margin=dict(l=50, r=50, t=50, b=50),
                  width=1200, height=600,)
fig.show()

#%% obtain sankeymatic text 
# Reverse the label_to_index dictionary to get index to label mapping
file='test69'
day = datetime(2001,1,8,0,0,0)
en = pf.comp_en_for_one_day(e[file], day)

flows, unique_labels, label_to_index = pf.make_sankey_flows(en, p[file])

index_to_label = {v: k for k, v in label_to_index.items()}

# Create the SANKEYMatic formatted strings
sankeymatic_lines = []
for key in flows:
    source_label = index_to_label[flows[key]['source']]
    target_label = index_to_label[flows[key]['target']]
    value = round(flows[key]['value'],2)
    sankeymatic_lines.append(f"{source_label} [{value}] {target_label}")

# Join all lines into a single string separated by new lines
sankeymatic_text = "\n".join(sankeymatic_lines)
print(sankeymatic_text)

#%% calculate KPIs
file = 'test69'
t1 = datetime(2001,7,16,0,0,0)
t2 = datetime(2001,7,17,0,0,0)
energy = e[file].loc[t1:t2]
el_em, gas_em, energy = pf.cal_emissions(energy)
print(f'electricty emission: {round(el_em,2)}, gas emission: {gas_em}')

#%% mass balance for SS buff load side pump
fig,ax = plt.subplots()
key='test51'
tf[key][['mpumpHPS_in','mpumpHPS_out']].plot(ax=ax, style='--')
ax0 = ax.twinx()
c[key]['ctr_hp'].plot.area(ax=ax0, alpha=0.2)
mb[key]['hp_source_pump'].plot(ax=ax0, color='black', style='-.')
ax.legend()
ax0.legend()
ax.set_xlim([t1,t2])

#%% print results
test = 'test55'
tx = datetime(2001,2,16,16,12,0)
tamb = tf[test]['Tamb'].loc[tx]
coll_in = tf[test]['Tcoll_in'].loc[tx]
t_in_below_amb = c[test]['t_inlet_below_ambient'].loc[tx]
coll_pump = c[test]['coll_pump'].loc[tx]
print(f"{test}: tamb: {tamb}, tcoll_in: {coll_in}, t_inlet_lelow_amb: {t_in_below_amb}, coll_pump: {coll_pump}")
#%%
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

# Create traces for each series
trace1 = go.Scatter(x=e['test44'].index, y=e['test44']['Qhp'],
                    mode='lines', fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)',  # Adjust color and transparency
                    name='test44')

trace2 = go.Scatter(x=e['test45'].index, y=e['test45']['Qhp'],
                    mode='lines', line=dict(width=4), name='test45')

# Create the figure and add the traces
fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)

# Update layout to add titles and labels
fig.update_layout(title='Qhp Plot', xaxis_title='Index', yaxis_title='Qhp',
                  legend_title='Tests', 
                  xaxis=dict(rangeslider=dict(visible=True), type='date'))

# Show the plot
fig.show()

e['test44']['Qhp'].plot.area(ax=ax,alpha=0.2,label='test44')
e['test45']['Qhp'].plot(ax=ax,linewidth=2,label='test45')
#%% check sims plot in plotly using rangeslider
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate random time series data
np.random.seed(42)
# dates = c[key].index
# tf = temp_flow.copy()
# e = energy.copy()
# c = controls.copy()
# Create a 4x2 subplot layout
fig = make_subplots(
    rows=4, cols=2,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Collector Panel", "Collector Control",
                    "Heat Pump", "Heat Pump Control",
                    "DHW Tank", "DHW Control",
                    "Space Heating", "Heating Control"),
    specs=[
        [{"secondary_y": True}, {"secondary_y": True}],
        [{"secondary_y": True}, {"secondary_y": True}],
        [{"secondary_y": True}, {"secondary_y": True}],
        [{"secondary_y": True}, {"secondary_y": True}]
    ]
)

# Add traces to each subplot
fig.add_trace(go.Scatter(x=tf.index, y=tf["Tcoll_in"], mode="lines", name="Tcoll_in"), row=1, col=1)
fig.add_trace(go.Scatter(x=tf.index, y=tf["Tcoll_out"], mode="lines", name="Tcoll_out"), row=1, col=1)
fig.add_trace(go.Scatter(x=tf.index, y=tf["Tamb"], mode="lines", name="Tamb"), row=1, col=1)
fig.add_trace(go.Scatter(x=tf.index, y=e["QuColl"], mode="lines", name="QuColl"), row=1, col=1, secondary_y=True)

fig.add_trace(go.Scatter(x=c.index, y=c["coll_pump"], mode="lines", name="coll_pump"), row=1, col=2)
fig.add_trace(go.Scatter(x=c.index, y=c["ctr_irr"], mode="lines", name="ctr_irr"), row=1, col=2)
fig.add_trace(go.Scatter(x=c.index, y=c["ctr_coll_t"], mode="lines", name="ctr_coll_t"), row=1, col=2)
fig.add_trace(go.Scatter(x=c.index, y=e["Qirr"], mode="lines", name="Qirr"), row=1, col=2, secondary_y=True)

# Update layout
fig.update_layout(
    height=800,
    showlegend=True,
    title="4x2 Subplots with Range Slider",
    xaxis_rangeslider_visible=True
)

# Show plot
fig.show()
#%%
t1 = datetime(2001,6,6, 0,0,0)
t2 = datetime(2001,6,12, 0,0,0)

fig, axs = plt.subplots(len(e), 1, figsize=(10, 2*len(e)))
axs0 = [ax.twinx() for ax in axs]
for i, label in enumerate(e):
    pt[label] = Plots(c[label],e[label],tf[label])
    e[label][['Qhp','Qrad1','Qrad2']].plot.area(ax=axs[i], alpha=0.2, stacked=False)
    # e[label][['Qrad1','Qrad2']].sum(axis=1).plot(ax=axs[i], label='Qsh_delivered')
    # c[label][['heatingctr1','heatingctr2']].plot(ax=axs[i],style='--', color=['lightskyblue','dodgerblue'])
    # tf[label]['unmet'].plot(ax=axs[i],style='--', color='black')
    tf[label][['Tfloor1','Tfloor1_2','Tfloor2']].plot(ax=axs0[i], 
                                          color=['mediumvioletred', 'darkmagenta','green'])
    tf[label][['T1op_1','T1op_2','T2op']].plot(ax=axs0[i],  style=':',
                                          color=['mediumvioletred', 'darkmagenta','green'])
    temp_flow[['Tset1','Tset2']].plot(ax=axs0[i], style='--', 
                                              color=['mediumvioletred', 'green'])
    # pf.plot_specs(axs[i], t1,t2, 0,5, 'energy[kWh]',legend_loc='upper left')
    # pf.plot_specs(axs0[i], t1,t2, -15,34, 'Temperature [deg C]',legend_loc='upper right')
    # e[label][['QuColl','Qheat','Qaux_dhw']].plot.area(ax=axs[i], alpha=0.2, stacked=False)
    # tf[label][['Tcoll_in','Tcoll_out','Tamb']].plot(ax=axs0[i])
    pf.plot_specs(axs[i], t1,t2, -2,6, 'energy[kWh]',legend_loc='upper left')
    pf.plot_specs(axs0[i], t1,t2, 10,30, 't',legend_loc='upper right')
    heating_demand = round(e[label]['Qheat'].sum()*0.1, 2)
    # unmet = round(tf[label]['unmet'].sum(),2)
    axs[i].set_title(label)
    print(label + '(heat) ='+ str(heating_demand) + #' kWh, unmet hours = '+str(unmet)+
          ' Qhp = '+str(round(e[label]['Qhp'].sum()*0.1,2)))

#%% QuColl cumulative plot
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
fig = make_subplots(specs=[[{"secondary_y": True}]])
for i, label in enumerate(e):
    color = colors[i % len(colors)]
    pt[label] = Plots(c[label],e[label],tf[label])
    
    fig.add_trace(go.Scatter(x=e[label].index, y=e[label]['Qcoll_cum'], 
                             mode='lines', name=label,
                             line=dict(color=color, dash='solid')))
    fig.add_trace(go.Scatter(x=e[label].index, y=e[label]['QuColl'],
                             mode='lines', name=label,
                             line=dict(color=color, dash='dot')), 
                  secondary_y=True)

fig.update_layout(title='Qcoll_cum and QuColl from Different DataFrames',
                  xaxis_title='Index',
                  yaxis_title='kWh')
fig.update_yaxes(title_text="QuColl Value", secondary_y=True)


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

#%% manual calculation of Qrad
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

#%% plotly rangeslider
import plotly.graph_objects as go
from plotly.subplots import make_subplots

file_index = int(''.join([i for i in file if i.isdigit()]))
plot_name = ('Design case: '+ inputs['design_case'].loc[file_index] 
             + ', Area:' + str(inputs['coll_area'].loc[file_index]) 
             + ', Volume:' + str(inputs['volume'].loc[file_index])
             + ', R level:' + inputs['r_level'].loc[file_index]
             + ', File index:' + str(file_index))

fig = make_subplots(          
            rows=4, cols=1, 
            subplot_titles=('DHW tank', 'HP', 'SH', 'Panel'),
            vertical_spacing=0.1, shared_xaxes=True,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], 
                   [{"secondary_y": True}], [{'secondary_y': True}]])


fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['T1_dhw'], name="T1_dhw",
                         line_color='orangered'), row=1, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tat_tap'], name="T@tap",
                         line_color='rgb(245,161,39)'), row=1, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['T6_dhw'], name="T6 dhw",
                         line_color='deepskyblue'), row=1, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['mdhw2tap'], name="mdhw2tap",
                         line=dict(color='black', width=0.7, dash='dash')),
              secondary_y=True, row=1, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Thp_load_out'], name='Thp_out_load',
                         line=dict(color='orangered', width=0.9)), row=2,col=1)
fig.add_trace(go.Scatter(x=controls.index, y=controls['hp_div'], name="hp_div",
                          line=dict(color='black', width=1, dash='dash')),
              secondary_y=True, row=2, col=1)
fig.add_trace(go.Scatter(x=energy.index, y=energy['Qhp'], name="Qhp",
                          fill='tozeroy',fillcolor='rgba(44,209,209,0.5)', mode='none'),
              secondary_y=True, row=2, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tfloor1'], name='Tfloor1',
                          line = dict(color='rgba(46,167,50,1)')), row=3, col=1)              
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tfloor2'], name='Tfloor2',
                          line = dict(color='rgba(255,131,3,1)')), row=3, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tset1'], name='Tset1',
                          line = dict(color='rgba(46,167,50,0.8)', dash='dash')), row=3, col=1)              
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tset2'], name='Tset2',
                          line = dict(color='rgba(255,131,3,0.8)', dash='dash')), row=3, col=1)

fig.add_trace(go.Scatter(x=energy.index, y=energy['Qrad1'], name="Qrad1",
                          mode='none', fill='tozeroy', fillcolor='rgba(50,168,82,0.5)'),
              secondary_y=True, row=3, col=1)
fig.add_trace(go.Scatter(x=energy.index, y=energy['Qrad2'], name="Qrad2",
                          mode='none', fill='tozeroy', fillcolor='rgba(255,131,3,0.5)'), 
              secondary_y=True, row=3, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_in'], name='Tcoll_in',
                         line=dict(color='deepskyblue',width=0.7)), row=4,col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_out'], name='Tcoll_out',
                         line=dict(color='orangered',width=0.7)), row=4,col=1)
fig.add_trace(go.Scatter(x=energy.index, y=energy['QuColl'], name='QuColl',
                         mode='none', fill='tozeroy', fillcolor='rgba(255,173,3,0.5)'),
              secondary_y=True, row=4, col=1)
fig.add_trace(go.Scatter(x=energy.index, y=energy['Qirr'], name='Qirr',
                         line=dict(color='rgb(245,161,39)', width=1)), 
              secondary_y=True, row=4,col=1)
fig.update_yaxes(range=[-50, 50], row=4, col=1)
fig.update_yaxes(range=[0, 2], secondary_y=True, row=4, col=1)

fig.update_layout(title_text=plot_name,
                  xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05,
                  height=1000)
fig.update_layout(yaxis2={'tickformat': ',.0'},)
fig.show()

#%% same plot in matplotlib
# file_index = int(''.join([i for i in file if i.isdigit()]))
# plot_name = ('Design case: '+ inputs['design_case'].loc[file_index] 
#               + ', Area:' + str(inputs['coll_area'].loc[file_index]) 
#               + ', Volume:' + str(inputs['volume'].loc[file_index])
#               + ', R level:' + inputs['r_level'].loc[file_index]
#               + ', File index:' + str(file_index))
plot_name = 'SH buffer, type 166 for SH tank hysteresis'
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2001,1,7, 22,0,0)
fig,(ax1,ax5,ax6, ax2,ax3,ax4) = plt.subplots(6,1, figsize=(19,9.8), sharex=True)

ax10 = ax1.twinx()
temp_flow[['T1_dhw','T6_dhw','Tat_tap']].plot(ax=ax1,linewidth=1,color=['firebrick','tab:blue','orange'])
temp_flow['mdhw2tap'].plot.area(ax=ax10, color='orange',alpha=0.4)
pf.plot_specs(ax1, t1,t2,0,80,ylabel='t', legend_loc='center left', title='DHW')
pf.plot_specs(ax10, t1,t2,0,300,ylabel='f', legend_loc='center right')

ax50 = ax5.twinx()
temp_flow[['T1_sh', 'T6_sh','Tsh_return']].plot(ax=ax5, color=['firebrick','tab:blue','blue'])
temp_flow['mrad1_in'].plot(ax=ax50, color='firebrick',linestyle='--')
temp_flow['mrad2_in'].plot.area(ax=ax50, color='tab:blue', alpha=0.2)
pf.plot_specs(ax5, t1,t2,0,90,ylabel='t', legend_loc='center left', title='SH')
pf.plot_specs(ax50, t1,t2,0,None,ylabel='f', legend_loc='center right')

ax20 = ax2.twinx()
temp_flow['Thp_load_out'].plot(ax=ax2, color='firebrick')
energy['Qhp'].plot.area(ax=ax20, alpha=0.3, color='tab:blue')
energy['Qaux_dhw'].plot.area(ax=ax20, color='grey',alpha=0.3)
controls['hp_div'].plot(ax=ax20, color='black', linestyle='--')
pf.plot_specs(ax2, t1,t2,0,120,ylabel='t', legend_loc='center left', title='HP')
pf.plot_specs(ax20, t1,t2,0,3.5,ylabel='p', legend_loc='center right')

ax30 = ax3.twinx()
temp_flow['mmixDHWout'].plot.area(ax=ax3, alpha=0.2)
temp_flow['msh_in'].plot(ax=ax3)
# temp_flow['msh_in'].plot.area(ax=ax3,alpha=0.2)
controls['pvt_load_loop'].plot(ax=ax30, linestyle='--')
pf.plot_specs(ax3, t1,t2,0,None,ylabel='f', legend_loc='center left', title='HP div')
pf.plot_specs(ax30, t1,t2,0,3,ylabel='controls', legend_loc='center right')

ax60 = ax6.twinx()
temp_flow[['Tfloor1','Tfloor2']].plot(ax=ax6, color=['firebrick','green'])
temp_flow[['Tset1','Tset2']].plot(ax=ax6, color=['firebrick','green'], linestyle='--')
energy[['Qrad1','Qrad2']].plot.area(ax=ax60, color=['firebrick','green'], alpha=0.3)
pf.plot_specs(ax6, t1,t2,0,30,ylabel='t', legend_loc='center left', title='SH')
pf.plot_specs(ax60, t1,t2,0,None,ylabel='p', legend_loc='center right')  

ax40=ax4.twinx()
temp_flow[['Tcoll_in','Tcoll_out']].plot(ax=ax4, color=['tab:blue','firebrick'])
energy['QuColl'].plot.area(ax=ax40, color='gold', alpha=0.3,stacked=False)
energy['Qirr'].plot(ax=ax40,color='gold')
pf.plot_specs(ax4, t1,t2,-25,100,ylabel='t', legend_loc='center left', title='Collector panel')
pf.plot_specs(ax40, t1,t2,None,None,ylabel='p', legend_loc='center right')  

fig.suptitle(plot_name)

#%% Heat pump energy balance
t1 = datetime(2001,2,9, 7,0,0)
t2 = datetime(2001,2,17, 22,0,0)
df = energy[t1:t2]
df = df.resample('H').sum()
fig,ax = plt.subplots()
df['ht_to_load'].plot(ax=ax, kind='bar',position=1, width=0.2,color='green')
df[['ht_from_source','Qhp']].plot(ax=ax, kind='bar',stacked=True,position=0, width=0.2, color=['gold','skyblue'])
ax.legend()

#%% 
from plotting_performance_data_ecoforest import plot_hp_performance
fig, ax, df = plot_hp_performance()

t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2002,1,1, 0,0,0)
for file in tf:
    tf[file],c[file] = pf.new_columns_for_map(tf[file], c[file])

file='test73'
temp_flow = tf[file]
controls = c[file]
xi = temp_flow['Thp_source_in']*controls['ctr_hp']
yi_dhw = (temp_flow['Thp_load_in']*(controls['div_load']==1)*controls['ctr_hp']).replace(0,np.nan)
yi_sh = (temp_flow['Thp_load_in']*(controls['div_load']==0)*controls['ctr_hp']).replace(0,np.nan)
plt.scatter(xi,yi_dhw, edgecolors= "green", facecolors='none', linewidth=0.6,label='DHW', alpha=0.3)
plt.scatter(xi,yi_sh, edgecolors= "red", facecolors='none', linewidth=0.6,label='SH', alpha=0.3)



#%% Scatter plot on the HP performance map
from plotting_performance_data import plot_performance_map
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2002,1,1, 0,0,0)
tf = temp_flow[t1:t2]
fig, ax = plot_performance_map()
plt.scatter(tf['Tcoll_out'],tf['thp_load_in_dhw'], edgecolors= "green", facecolors='none', linewidth=0.6)
plt.scatter(tf['Tcoll_out'],tf['thp_load_in_dhw'], edgecolors= "green", facecolors='none', linewidth=0.6,label='DHW')
plt.scatter(tf['Tcoll_out'],tf['thp_load_in_sh'], edgecolors= "red", facecolors='none', linewidth=0.6,label='SH')
plt.legend()

import plotly.graph_objects as go

# Assuming tf is your DataFrame
fig = go.Figure(data=go.Scatter(
    x=tf['Tcoll_out'],
    y=tf['Thp_load_in'],
    mode='markers',
    marker=dict(color='rgba(135, 206, 250, 0.8)', size=10, line=dict(width=0.6, color='black')),
    text=tf.index,  # This will show the index of the DataFrame on hover
    hoverinfo='text+x+y'  # Customizes the hover text
))

fig.update_layout(
    title='Scatter plot of Tcoll_out vs. Thp_load_in',
    xaxis_title='Tcoll_out',
    yaxis_title='Thp_load_in',
    plot_bgcolor='white'
)

fig.show()

#%% collector and HP plotly plot
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'

fig = make_subplots(          
            rows=3, cols=1, 
            subplot_titles=('DHW tank', 'HP', 'SH', 'Panel'),
            vertical_spacing=0.1, shared_xaxes=True,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])


fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_out'], name="Tcoll_out",
                         line_color='orangered'), row=1, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tcoll_in'], name="Tcoll_in",
                         line_color='yellow'), row=1, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tamb'], name="Tamb",
                          line_color='rgb(245,161,39)'), row=1, col=1)

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['m_coll'], name="m_coll",
                         line=dict(color='black', width=0.7, dash='dash')),
              secondary_y=True, row=1, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['mrad1_in'], name="mrad2",
                         line=dict(color='green', width=0.7, dash='dash')),
              secondary_y=True, row=1, col=1)

fig.add_trace(go.Scatter(x=controls.index, y=controls['coll_pump'], name="coll_pump",
                          fill='tozeroy',fillcolor='rgba(44,209,209,0.5)', mode='none'),
              secondary_y=True, row=2, col=1)
fig.add_trace(go.Scatter(x=controls.index, y=controls['ctr_irr'], name="ctr_irr",
                          line_color='rgb(245,161,39)'), row=2, col=1)
fig.add_trace(go.Scatter(x=controls.index, y=controls['ctr_dhw'], name="ctr_dhw",
                          line_color='blue'), row=2, col=1)
fig.add_trace(go.Scatter(x=controls.index, y=controls['ctr_sh'], name="ctr_sh",
                          line_color='orangered'), row=2, col=1)

fig.add_trace(go.Scatter(x=energy.index, y=energy['COP'], name="COP",
                          fill='tozeroy',fillcolor='rgba(44,209,209,0.5)', mode='none'),
              secondary_y=True, row=3, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Thp_source_in'], name="Thp_source_in",
                          line_color='rgb(245,161,39)'), row=3, col=1)
fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Thp_load_in'], name="Thp_load_in",
                          line_color='blue'), row=3, col=1)
              
fig.update_layout(title_text='Test',
                  xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05,
                  height=1000)

#%% check if controls are calculated correctly
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2001,1,7, 0,0,0)
ctr = controls.astype(int)[t1:t2]
coll_pump = ctr['ctr_irr'] | ctr['ctr_dhw'] | ctr['ctr_sh']
res = ctr['coll_pump'].compare(coll_pump)

fig,ax= plt.subplots()
ctr['coll_pump'].plot.area(ax=ax,alpha=0.2)
coll_pump.plot(ax=ax)

#%% new columns to segrregate Tcond_in into dhw and sh columns
temp_flow['thp_load_in_dhw'] = temp_flow['Thp_load_in']*(controls['ctr_dhw'])
controls['no_dhw'] = controls['ctr_dhw'].apply(lambda x: 1 if x == 0 else 0)
temp_flow['thp_load_in_sh'] = temp_flow['Thp_load_in']*(controls['no_dhw'])
temp_flow['thp_load_in_dhw'].replace(0,np.NaN, inplace=True)
temp_flow['thp_load_in_sh'].replace(0,np.NaN, inplace=True)

#%% Calculate base case: demand vs outdoor temp, annual energy demand 
#   (cp_PV: case 1047, WWHP with ST: case 1089; ASHP: case 1049)
qheat_sh = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']
Qheat_sh = qheat_sh.resample('D').sum()*0.1

avg_temp = temp_flow['Tamb'].resample('D').mean()[:-1]
Qheat_tot = (energy['Qheat'].resample('D').sum()*0.1)[:-1]
df = pd.concat([avg_temp,Qheat_tot], axis=1)

fig,ax = plt.subplots()
ax.scatter(df['Tamb'],df['Qheat'],s=20, edgecolors= "darkred", facecolors='none', linewidth=0.6)
pf.plot_specs(ax,xlabel='Avg outdoor temp [deg C]', ylabel='Total heat demand [kWh]', 
              title='Heat signature of the base case', ygrid=True)

test = df[(df['Tamb']<0) & (df['Qheat']>80)]

#%% Compare base case, WWHP and ASHP from cases 1047_cp, 1089, 1059 resp.
e,tf = {},{}
for i in e1:
    e[i] = (e1[i].resample('D').sum()*0.1)[:-1]
    tf[i] = tf1[i].resample('D').mean()[:-1]
    
df = pd.concat([tf['1047_cp']['Tamb'],
                e['1047_cp']['Qheat'],
                e['1089']['Qhp_heating_out'],
                e['1059']['Qhp_heating_out']],axis=1,
               keys = ['Tamb',"Heat demand","WWHP",'ASHP' ])
df2 = df.sort_values(by='Tamb')

fig,ax = plt.subplots()
ax.plot(df2['Tamb'],df2['WWHP'],label='WWHP')
ax.plot(df2['Tamb'],df2['ASHP'],label='ASHP')
ax.plot(df2['Tamb'],df2['Heat demand'],'--',label='Bldg demand')
ax.legend()

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