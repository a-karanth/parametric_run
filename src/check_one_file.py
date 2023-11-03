# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:37:03 2023

@author: 20181270
"""

import sys
import time                 # to measure the computation time
import os 
dir_trial = os.getcwd()
#os.chdir('..')
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
file = '840_cp'
prefix = directory + folder + file

t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

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

energy_monthly, energy_annual = pf.cal_integrals(energy)

el_bill, gas_bill = pf.cal_costs(energy)
el_em, gas_em = pf.cal_emissions(energy)
pl,pe = pf.peak_load(energy)
rldc,ldc = pf.cal_ldc(energy)
opp_im, opp_ex, import_in, export_in = pf.cal_opp(rldc)
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
