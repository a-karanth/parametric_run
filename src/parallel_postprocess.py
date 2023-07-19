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
# from ModifyType56 import ModifyType56
# from PostprocessFunctions import PostprocessFunctions as pf
# from Plots import Plots
# from PlotGroups import PlotGroups
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
from SALib.analyze import sobol, morris
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

global directory, result_folder
directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\\PhD\\TRNSYS\\Publication1\\pub_1\\src'
res_folder = 'res'
trn_folder = '\\res\\trn'
print('start')

# %%

# result_folder = '\\with_summer_loop\\2parameterSA_volume_area'

temp = os.listdir(directory+trn_folder)
labels = []
for i in temp:
    if '_temp_flow' in i:
        prefix = i[:-14]
    elif '_control_signal' in i:
        prefix = i[:-19]
    elif '_energy' in i:
        prefix = i[:-11]
    else:
        continue
    labels.append(prefix)

labels = np.array(labels)
labels = np.unique(labels)
labels = np.arange(86)
# labels = list(labels)
labels= labels.astype(str)
DT = '6min'


t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

#%%
os.chdir(directory)
from PostprocessFunctions import PostprocessFunctions as pf
pp = pf(DT)
def parallel_pp(label):
    
    
    if 'cp' in label:
        controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pf.cal_base_case(label)
        
    else:
        temp_flow = pd.read_csv(directory+trn_folder+'\\'+label+'_temp_flow.txt', delimiter=",",index_col=0)
        energy = pd.read_csv(directory+trn_folder+'\\'+label+'_energy.txt', delimiter=",", index_col=0)
        controls = pd.read_csv(directory+trn_folder+'\\'+label+'_control_signal.txt', delimiter=",",index_col=0)
        
        controls = pp.modify_df(controls, t_start, t_end)
        temp_flow = pp.modify_df(temp_flow, t_start, t_end)
        energy = pp.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        energy = pp.cal_energy(energy, controls)
        energy_monthly, energy_annual = pp.cal_integrals(energy)
        
    el_bill, gas_bill = pf.cal_costs(energy)
    el_em, gas_em = pf.cal_emissions(energy)
    print(label)
    return el_bill, gas_bill, el_em, gas_em, energy_annual, label

#%% 
# os.chdir(directory + result_folder)
# results = pd.DataFrame(columns=['el_bill','gas_bill', 'el_em', 'gas_em','energy_annual','lab'])
# count = 0
# for i in labels:
#     el_bill, gas_bill, el_em, gas_em, energy_annual, lab = parallel_pp(str(i))
#     results.loc[i] = el_bill, gas_bill, el_em, gas_em, energy_annual, lab
#     count = count+1
#     print(count)

# results['total_costs'] = results['el_bill']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000
# results.to_csv('sim_results'+'.csv', index=True)

#%%
# # multiprocessing that works
# t1 = time.time()
# if __name__ == "__main__":
#     pool = mp.Pool(8)
#     results = pool.map(parallel_pp, labels)
    
#     pool.close()
#     pool.join()
#     # results = []

#     # for i in range(len(labels)):
#         # time.sleep(3)  # Delay of 15 seconds
#         # result = pool.apply_async(parallel_pp, (str(i),))
#         # results.append(result)
#     # output = [result.get() for result in results]
    
#     # Wait for the multiprocessing tasks to complete
#     # for result in results:
#     #     result.get()
        
# t2 = time.time()
# print(t2-t1)

#%% exporting the results in a csv
# output = pd.DataFrame(columns=['el_bill','gas_bill', 'el_em', 'gas_em','energy_annual','lab'])
# for i in range(len(results)):
#     output.loc[i] = results[i]
# output.to_csv(res_folder+'\\'+'sim_results'+'.csv', index=True)

#%%
results = pd.read_csv(res_folder+ '\\'+'sim_results.csv', index_col=0)
results['total_costs'] = results['el_bill']+results['gas_bill']
results['total_emission'] = (results['el_em']+results['gas_em'])/1000
existing = pd.read_csv('res\\trn\\list_of_inputs.csv',header=0)
dfresults = pd.concat([existing, results],axis=1)


dfmorris_st = pd.read_csv('res\\morris_st_sample.csv')
morris_out_st = pd.merge(dfmorris_st, dfresults, on = ['volume','coll_area','flow_rate','design_case','r_level'], how = 'left')
dfmorris_pvt = pd.read_csv('res\\morris_pvt_sample.csv')
morris_out_pvt = pd.merge(dfmorris_pvt, dfresults, on = ['volume','coll_area','flow_rate','design_case','r_level'], how = 'left')


sample_st= dfmorris_st.copy()
sample_st.drop(columns=['design_case'], inplace=True)
sample_pvt= dfmorris_pvt.copy()
sample_pvt.drop(columns=['design_case'], inplace=True)

#%% recreating problem - copied from sobol_method
input_st = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200],
              'r_level': ['r0','r1']}

input_pvt = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200],
              'r_level': ['r0','r1']}

def cal_bounds_scenarios(dct):
    key = list(dct.keys())
    bounds = []
    nscenarios = 1
    for i in key:
        bounds.append([0, len(input_st[i])-1])
        nscenarios = nscenarios*len(input_st[i])
    return bounds, nscenarios

#%% Creating bounds and scenarios
bounds_st, nscenarios_st = cal_bounds_scenarios(input_st)
bounds_pvt, nscenarios_pvt = cal_bounds_scenarios(input_pvt)

#%% Creating problems
problem_st = {
    'num_vars': len(input_st),
    'names':list(input_st.keys()),
    'bounds':bounds_st}

problem_pvt = {
    'num_vars': len(input_pvt),
    'names':list(input_pvt.keys()),
    'bounds':bounds_pvt}

#%% analysing morris samples
Si = morris.analyze(
    problem_pvt,
    sample_pvt,
    np.ravel(morris_out_pvt['total_costs']),
    conf_level=0.95,
    print_to_console=True,
    num_levels=4,
    num_resamples=100,
)

# dfsobol = pd.read_csv('morris_pvt_sample.csv')
# dfresults = results.copy()

# sobol_out = pd.merge(dfsobol, dfresults, on = ['volume','coll_area','design_case'], how = 'left')

# list_volume = [0.1, 0.2, 0.3, 0.4]
# list_coll_area = [8, 16, 20]
# list_design_case_st = ['cp','cp_PV', 'ST']
# list_design_case_pvt = ['cp','cp_PV', 'PVT']
# list_design_case_pvt_batt = ['PVT_Batt_6', 'PVT_Batt_9']
# list_flow = [50, 100, 200]

# problem_pvt = {
#     'num_vars': 4,
#     'names':['tank_volume', 'coll_area', 'design_case', 'flow'],
#     'bounds':[[0, len(list_volume)-1],
#               [0, len(list_coll_area)-1],
#               [0, len(list_design_case_pvt)-1],
#               [0, len(list_flow)-1],]}

# list_volume = [0.1, 0.15, 0.2, 0.25, 0.3]
# list_coll_area = [10, 12, 14, 16, 18, 20]
# list_design_case = ['base','base_PV', 'ST', 'PVT', 'PVT_Batt_6', 'PVT_Batt_9']
# problem = {
#     'num_vars': 3,
#     'names':['tank_volume', 'coll_area', 'design_case'],#, 'r_values'],
#     'bounds':[[0, len(list_volume)-1],
#               [0, len(list_coll_area)-1],
#               [0, len(list_design_case)-1],]}

# Si_cost = morris.analyze(problem, np.ravel(sobol_out['total_costs']), calc_second_order=True)
# Si_em = sobol.analyze(problem, np.ravel(sobol_out['total_emission']), calc_second_order=True)

# Si_cost.plot()
# plt.suptitle('KPI: Energy Cost')
# Si_em.plot()
# plt.suptitle('KPI: Emissions')