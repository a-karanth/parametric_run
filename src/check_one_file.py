# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:37:03 2023

@author: 20181270
"""

import sys
import time                 # to measure the computation time
import os 
dir_trial = os.getcwd()
os.chdir('..')
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
file = '197_cp'
prefix = directory + folder + file

# controls, energy, temp_flow = pf.create_dfs(prefix)   #use for regular
controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pf.cal_base_case(prefix) # use for base case
energy_monthly, energy_annual = pf.cal_integrals(energy)
el_bill, gas_bill = pf.cal_costs(energy)

#%% plots
pt = Plots(controls, energy, temp_flow)
t1 = datetime(2001,1,1, 0,0,0)
t2 = datetime(2001,1,8, 0,0,0)
pt.plot_q(t1, t2)

#%% long method

# temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter = ",", index_col=0)
# energy = pd.read_csv(prefix+'_energy.txt', delimiter = ",", index_col=0)
# controls = pd.read_csv(prefix+'_control_signal.txt', delimiter = ",", index_col=0)

# controls = pf.modify_df(controls, t_start, t_end)
# temp_flow = pf.modify_df(temp_flow, t_start, t_end)
# energy = pf.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
# energy = pf.cal_energy(energy, controls)

# energy['Qheat'] = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']
# energy['Qhp4sh'] = energy['Qheat']/2.8
# energy['Qhp4tank'] = energy['Qaux_dhw']/1.9
# energy['Qaux_dhw'] = 0
# energy['Qload'] = energy['Qdev']+energy['Qltg']+energy['Qhp4sh']+energy['Qhp4tank']
# energy['Qfrom_grid'] = energy['Qload']

