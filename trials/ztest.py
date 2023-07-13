# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:33:04 2023

@author: 20181270
"""
import subprocess           # to run the TRNSYS simulation
import shutil               # to duplicate the output txt file
import time                 # to measure the computation time
from ModifyType56 import ModifyType56
from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots
import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

# folder_name = '\\with_summer_loop\\2parameterSA_volume_area'
# folder_name = '\\Tests\\Ventilation'
folder_name = '\\Master'
# prefix = 'base_V0_25_A16'
# prefix = 'inf_0_5'
prefix = 'x'
destination = 'Diagrams/'
fig_prefix = '1min'
dt = '6min'

directory = (os.path.dirname(os.path.realpath(__file__)))
# mod56 = ModifyType56()
# mod56.change_r('base')
os.chdir(directory + folder_name)

fig_format = '.svg'

t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

pp = pf(dt)

# temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter = ",", index_col=0)
# energy = pd.read_csv(prefix+'_energy.txt', delimiter = ",", index_col=0)
# controls = pd.read_csv(prefix+'_control_signal.txt', delimiter = ",", index_col=0)

# controls = pp.modify_df(controls, t_start, t_end)
# temp_flow = pp.modify_df(temp_flow, t_start, t_end)
# energy = pp.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
# energy = pp.cal_energy(energy, controls)

# energy_monthly, energy_annual = pf.cal_integrals(energy)

# controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pp.cal_base_case(prefix)

t1 = datetime(2001,2,17, 0,0,0)
t2 = datetime(2001,2,26, 0,0,0)

prefix_loop = [prefix]

# print('prefix = '+prefix)

if 'cp' in prefix:
    controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pf.cal_base_case(prefix)
else:
    temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter = ",", index_col=0)
    energy = pd.read_csv(prefix+'_energy.txt', delimiter = ",", index_col=0)
    controls = pd.read_csv(prefix+'_control_signal.txt', delimiter = ",", index_col=0)

    controls = pp.modify_df(controls, t_start, t_end)
    temp_flow = pp.modify_df(temp_flow, t_start, t_end)
    energy = pp.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
    energy = pp.cal_energy(energy, controls)
    
    energy_monthly, energy_annual = pf.cal_integrals(energy)
    
pt = Plots(controls, energy, temp_flow)

pt.plot_controls_cp(t1,t2)
plt.suptitle('plot_controls ' + prefix)

energy_annual[['Qhp4sh','Qhp4tank']].plot.bar(stacked=True)
plt.suptitle(prefix)

# el_bill, gas_bill = pf.cal_costs(energy)

# energy['Q2batt'] = [(i) if i>0 else 0 for i in energy['Qbatt_tofrom'] ]
# energy['Qfrombatt'] = [(i) if i<0 else 0 for i in energy['Qbatt_tofrom'] ]
# print(energy['Q2batt'].sum()*5/60)
# print(energy['Qfrombatt'].sum()*5/60)
# print('Total load: ' + str(energy_annual['Qload'][0]) + ' kWh')
# print('Total import: ' + str(energy_annual['Qfrom_grid'][0]) + ' kWh')

# pp.summarize_results(controls, energy, temp_flow)
# heat = energy['Qhp4sh'].sum()*0.1
# dhw = energy['Qhp4tank'].sum()*0.1
# print('heating = '+ str(heat))
# print('Aux for DHW = '+ str(dhw))
# print('Total gas = ' + str((heat+dhw)/0.98))

# for i in np.arange(1,13):
#     pt.plot_wea(temp_flow, energy, i)

#%% trial of all Plot functions to test backtracing
'''
fig,ax = plt.subplots()
pt.select_q(ax, t1,t2, 'Qhp', 'Q2grid')
fig.suptitle('select_q')

fig2,ax2 = plt.subplots()
pt.plot_hp(ax2, t1,t2)
fig2.suptitle('plot_hp')

fig3, ax3 = plt.subplots()
pt.plot_dhw(ax3, t1,t2)
fig3.suptitle('plot_dhw')

fig4, ax4 = plt.subplots()
pt.plot_dhw_aux(ax4, t1, t2)
fig4.suptitle('plot_dhw_aux')

pt.plot_sh_dhw(t1,t2)
plt.suptitle('plot_dhw')

pt.plot_q(t1,t2)
plt.suptitle('plot_q')

pt.plot_controls(t1,t2)
plt.suptitle('plot_controls')

pt.plot_sh(t1,t2)
plt.suptitle('plot_sh')

pt.plot_batt(prefix,t1,t2)
plt.suptitle('plot_batt')

pt.plot_summer_loop(t1,t2)
plt.suptitle('plot_controls')

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(16,9))
pt.plot_colormap(ax1, 'SOC','SOC', 'gist_earth_r', 0.09,0.9)
pt.plot_colormap(ax2, 'Q2grid', 'Export [kWh]', 'PRGn', 0,1.8)
pt.plot_colormap(ax3, 'Tavg_dhw', 'Avg tank temp [deg C]', 'gist_heat', 20,75)
fig.suptitle(prefix + ' Colormaps')
'''

