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
file = 'test10'
prefix = directory + folder + file

inputs = pd.read_csv(folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
t_start = datetime(2001,2,9, 0,0,0)
t_end = datetime(2001,2,17, 0,0,0)

#%% Read files
if 'cp' in file:
    controls, energy, temp_flow = pf.cal_base_case(prefix)
    
else:
    temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter=",",index_col=0)
    energy = pd.read_csv(prefix+'_energy.txt', delimiter=",", index_col=0)
    controls = pd.read_csv(prefix+'_control_signal.txt', delimiter=",",index_col=0)
    
    controls = pf.modify_df(controls)#, t_start, t_end)
    temp_flow = pf.modify_df(temp_flow)#, t_start, t_end)
    energy = pf.modify_df(energy)/3600#, t_start, t_end)/3600     # kJ/hr to kW 
    energy = pf.cal_energy(energy, controls)

# occ = pd.read_csv(directory+folder+'occ.txt', delimiter=",",index_col=0)
# occ = pf.modify_df(occ, t_start, t_end)
# controls = pd.concat([controls,occ],axis=1)

# temp_flow = pf.unmet_hours(controls, temp_flow)

# energy_monthly, energy_annual = pf.cal_integrals(energy)

# el_bill, gas_bill = pf.cal_costs(energy)
# el_em, gas_em = pf.cal_emissions(energy)
# pl,pe = pf.peak_load(energy)
# rldc,ldc = pf.cal_ldc(energy)
# opp_im, opp_ex, import_in, export_in = pf.cal_opp(rldc)
# cop = pf.cal_cop(energy)
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
occ = pf.modify_df(occ)#, t_start, t_end)
#%% run multiple files
files = [#'2000',
         '2001', '2002', '2003', '2004', '2005', '2006', #'2007',
         '2008', '2009', '2010', '2011', '2012', '2013']

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
files = ['2008', '2009', '2010', '2011', '2012', '2013']
c = {key: c1[key] for key in files}
e = {key: e1[key] for key in files}
tf = {key: tf1[key] for key in files}
#%%
t1 = datetime(2001,1,11, 0,0,0)
t2 = datetime(2001,1,12, 0,0,0)

fig, axs = plt.subplots(len(e), 1, figsize=(10, 2*len(e)))
axs0 = [ax.twinx() for ax in axs]
for i, label in enumerate(e):
    pt = Plots(c[label],e[label],tf[label])
    e[label]['Qhp'].plot.area(ax=axs[i], alpha=0.2, stacked=False)
    # e[label][['Qheat_living1','Qheat_living2']].sum(axis=1).plot(ax=axs[i], label='Qheat')
    # c[label][['heatingctr1','heatingctr2']].plot(ax=axs[i],style='--', color=['lightskyblue','dodgerblue'])
    # tf[label]['unmet'].plot(ax=axs[i],style='--', color='black')
    # tf[label][['Tfloor1','Tfloor2','Thp_load_out']].plot(ax=axs0[i], 
    #                                       color=['mediumvioletred', 'darkmagenta','green'])
    # temp_flow[['Tset1','Tset2']].plot(ax=axs0[i], style='--', 
    #                                           color=['mediumvioletred', 'darkmagenta'])
    # pf.plot_specs(axs[i], t1,t2, 0,5, 'energy[kWh]',legend_loc='upper left')
    # pf.plot_specs(axs0[i], t1,t2, -15,34, 'Temperature [deg C]',legend_loc='upper right')
    e[label][['QuColl','Qheat','Qaux_dhw']].plot.area(ax=axs[i], alpha=0.2, stacked=False)
    tf[label]['Tcoll_out'].plot(ax=axs0[i])
    pf.plot_specs(axs[i], t1,t2, 0,6, 'energy[kWh]',legend_loc='upper left')
    pf.plot_specs(axs0[i], t1,t2, None,None, 't',legend_loc='upper right')
    heating_demand = round(e[label]['Qheat'].sum()*0.1, 2)
    unmet = round(tf[label]['unmet'].sum(),2)
    axs[i].set_title(label)
    print(label + '(heat) ='+ str(heating_demand) + #' kWh, unmet hours = '+str(unmet)+
          ' Qhp = '+str(round(e[label]['Qhp'].sum()*0.1,2)))

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

#%% pvt load loop calculation
t1 = datetime(2001,2,12, 7,0,0)
t2 = datetime(2001,2,12, 22,0,0)
fig, (ax2,ax0,ax, ax4) = plt.subplots(4,1)

temp_flow[['Tcoll_out','Tcoll_in']].plot(ax=ax2, color=['firebrick','tab:blue'])
pf.plot_specs(ax2, t1,t2,None,None,ylabel='t', legend_loc='center left', title='Collector panel')

temp_flow['m_coll'].plot(ax=ax0)
ax00 = ax0.twinx()
energy['QuColl'].plot(ax=ax00, color='gold')
pf.plot_specs(ax00, t1,t2,-0.2,None,ylabel='Energy', legend_loc='center right')

controls['coll_pump'].plot(ax=ax)
controls['pvt_load_loop'].plot.area(ax=ax,alpha=0.2)
pf.plot_specs(ax0, t1,t2,None,None,ylabel='f', legend_loc='center left')
pf.plot_specs(ax, t1,t2,None,None,ylabel='p', legend_loc='center right')  

controls['ctr_irr'].plot.area(ax=ax4, color='gold', alpha=0.2)
controls['ctr_sh'].plot(ax=ax4, marker='*')
controls['ctr_dhw'].plot(ax=ax4, linestyle='--')
pf.plot_specs(ax4, t1,t2,None,None,ylabel='controls', legend_loc='center right')

fig.suptitle(plot_name)
