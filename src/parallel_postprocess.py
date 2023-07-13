# %%
# -*- coding: utf-8 -*-
"""
post processing results using parallel processing

@author: 20181270
"""

import subprocess           # to run the TRNSYS simulation
import shutil               # to duplicate the output txt file
import time                 # to measure the computation time
import os 
import multiprocessing as mp
from ModifyType56 import ModifyType56
from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots
from PlotGroups import PlotGroups
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
from SALib.analyze import sobol
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

global directory, folder
directory = (os.path.dirname(os.path.realpath(__file__)))


# %%

result_folder = '\\with_summer_loop\\2parameterSA_volume_area'
os.chdir(directory+result_folder)
test = os.listdir(directory+result_folder)
labels = []
for i in test:
    if '_temp_flow' in i:
        prefix = i[:-14]
    elif '_control_signal' in i:
        prefix = i[:-19]
    elif '_energy' in i:
        prefix = i[:-11]
    labels.append(prefix)

labels = np.array(labels)
labels = np.unique(labels)

DT = '6min'
pp = pf(DT)

t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

def parallel_pp(label):
    if 'base' in label:
        controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pf.cal_base_case(label)
        
    else:
        temp_flow = pd.read_csv(label+'_temp_flow.txt', delimiter=",",index_col=0)
        energy = pd.read_csv(label+'_energy.txt', delimiter=",", index_col=0)
        controls = pd.read_csv(label+'_control_signal.txt', delimiter=",",index_col=0)
        
        controls = pp.modify_df(controls, t_start, t_end)
        temp_flow = pp.modify_df(temp_flow, t_start, t_end)
        energy = pp.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        energy = pp.cal_energy(energy, controls)
        energy_monthly, energy_annual = pp.cal_integrals(energy)
        
    el_bill, gas_bill = pf.cal_costs(energy)
    el_em, gas_em = pf.cal_emissions(energy)
    
    return el_bill, gas_bill, el_em, gas_em, energy_annual


results = pd.DataFrame(columns=['el_bill','gas_bill', 'el_em', 'gas_em','energy_annual'])
count = 0
# for i in labels:
#     el_bill, gas_bill, el_em, gas_em, energy_annual = parallel_pp(i)
#     results.loc[i] = el_bill, gas_bill, el_em, gas_em, energy_annual
#     count = count+1
#     print(count)

# results['total_costs'] = results['el_bill']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000
# results.to_csv('sim_results'+'.csv', index=True)

results = pd.read_csv('sim_results.csv', index_col=0)
results['volume'] = [float(label[label.find('_V')+2:label.find('_A')].replace('_','.')) for label in results.index]
results['coll_area'] = [int(label[label.find('_A')+2:]) for label in results.index]
results['design_case'] = [label[:label.find('_V')] for label in results.index]

dfsobol = pd.read_csv('Sobol_samples2.csv')
dfresults = results.copy()

sobol_out = pd.merge(dfsobol, dfresults, on = ['volume','coll_area','design_case'], how = 'left')

list_volume = [0.1, 0.15, 0.2, 0.25, 0.3]
list_coll_area = [10, 12, 14, 16, 18, 20]
list_design_case = ['base','base_PV', 'ST', 'PVT', 'PVT_Batt_6', 'PVT_Batt_9']
problem = {
    'num_vars': 3,
    'names':['tank_volume', 'coll_area', 'design_case'],#, 'r_values'],
    'bounds':[[0, len(list_volume)-1],
              [0, len(list_coll_area)-1],
              [0, len(list_design_case)-1],]}
Si_cost = sobol.analyze(problem, np.ravel(sobol_out['total_costs']), calc_second_order=True)
Si_em = sobol.analyze(problem, np.ravel(sobol_out['total_emission']), calc_second_order=True)

Si_cost.plot()
plt.suptitle('KPI: Energy Cost')
Si_em.plot()
plt.suptitle('KPI: Emissions')