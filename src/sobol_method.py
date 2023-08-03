# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:49:51 2023

@author: 20181270
"""

import os
import sys
import pkgutil
from warnings import warn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


directory = (os.path.dirname(os.path.realpath(__file__))) + '\\'
res_folder = 'res\\'
trn_folder = 'res\\trn\\'

#create one list for each variable indicating their input values
input_st = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200]} #,
             # 'r_level': ['r0','r1']}

input_pvt = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200],
              'r_level': ['r0','r1']}


list_design_case_pvt_batt = ['PVT_Batt_6', 'PVT_Batt_9']

#%% Function for creating bounds and scenarios
def cal_bounds_scenarios(dct):
    #   define problem to be analysed (number, name and range of the variables)
    #   creating the problem based on list index numbers. Which will then be used to 
    #   create final sample by applying to the lists declared above
    #   dct: dictionary of inputs to be given for simulation
    key = list(dct.keys())
    bounds = []
    nscenarios = 1
    for i in key:
        bounds.append([0, len(input_st[i])-1])
        nscenarios = nscenarios*len(input_st[i])
    return bounds, nscenarios

#%% Function to assign index values to the samples
def assign_indices(samples, inp, key_name, values):
    # samples: sample created by morris or sobol classes
    # inp: dictionary if inputs containing the different parameters
    # key_name: adding columns that are common to all rows, that need not be in the SA. eg: design_case
    # values: the value to be filled in each row of these columns. eg: ST
    keys = list(inp.keys())
    result = pd.DataFrame(columns=keys)
    for row in np.arange(len(samples)):
        result.loc[row] = samples[row]
        for i,k in zip(samples[row], keys):
            result.loc[row,k] = inp[k][i]
    result.columns = keys
    for i,j in zip(key_name,values):
        result[i] = j
    return result

#%% function for preparing SA variables
from SALib.sample import sobol, morris

def prepare_sa(sa_type, ip, key_names, values, N=2, prnt=False, prefix='x'):
    #   Creating of bounds and number of scenarios
    bounds, nscenarios = cal_bounds_scenarios(ip)
    
    #   Creation of problem dictionary
    problem = {
        'num_vars': len(ip),
        'names':list(ip.keys()),
        'bounds':bounds}
    
    #   Create the sample by defining the N value. samples contain index values for 
    #   the parameter lists. As the calculation for second order is defined as false, 
    #   the following equation will be used to defined the sample size: N∗(D+2) .
    #    As the values defined by Sobol and Morris are floats, they need to be rounded 
    #   and converted to int in order to be used as indices
    match sa_type:
        case 'Sobol':
            samp = sobol.sample(problem, N, calc_second_order=True)
        case 'Morris':
            samp = morris.sample(problem, N, num_levels=4, optimal_trajectories=None)
        
    samp = samp.round().astype(int)
    samp = assign_indices(samp, ip, key_names, values)
    if prnt:
        samp.to_csv(sa_type+'_samples_'+ prefix +'.csv', index=False)
    return problem, samp

#%% function to perform SA on the generated samples
from SALib.analyze import sobol as sobol_ana
from SALib.analyze import morris as morris_ana

def perform_sa(sa_type, kpi, problem, sample, sim_results, columns2drop, to_number=None):
    df = pd.merge(sample, sim_results, how='left')
    missing = df[np.isnan(df['el_bill'])]
    X = sample.drop(columns2drop, axis=1)
    if to_number:
        X[to_number] = (X[to_number].str.extract('(\d+)'))
    X = X.to_numpy(dtype=float)
    Y = df[kpi].ravel()             #to flatten series into a numpy array
    if len(missing != 0):
        return False, missing
    else:
        match sa_type:
            case 'Sobol':
                Si = sobol_ana.analyze(problem, 
                                   Y, 
                                   calc_second_order=True, 
                                   conf_level=0.95, 
                                   print_to_console=True)
            case 'Morris':
                Si = morris_ana.analyze(problem,
                                    X,
                                    Y,
                                    conf_level=0.95,
                                    print_to_console=True,
                                    num_levels=4,
                                    num_resamples=100,)
    return Si, missing

#%% collecting results
results = pd.read_csv(res_folder+'sim_results.csv', index_col='label')
results['total_costs'] = results['el_bill']+results['gas_bill']
results['total_emission'] = (results['el_em']+results['gas_em'])/1000

existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
dfresults = pd.concat([existing, results],axis=1)

#%% Running a loop to find samples with all existing data
count = 0
while True:
    print(1)
    problem, samp = prepare_sa('Sobol', input_st, ['design_case', 'r_level'], ['ST', 'r0'], N=4)
    Si, missing = perform_sa('Sobol', 'el_bill', problem, samp, dfresults, ['design_case','r_level'])
    count = count + 1
    if len(missing) == 0:
        Si.plot()
        break
#%%
# Si.plot()


#%% Breakdown of all steps
# #%%% Creating bounds and scenarios
# bounds_st, nscenarios_st = cal_bounds_scenarios(input_st)
# bounds_pvt, nscenarios_pvt = cal_bounds_scenarios(input_pvt)

# #%%% Creating problems
# problem_st = {
#     'num_vars': len(input_st),
#     'names':list(input_st.keys()),
#     'bounds':bounds_st}

# problem_pvt = {
#     'num_vars': len(input_pvt),
#     'names':list(input_pvt.keys()),
#     'bounds':bounds_pvt}


# #%%% Create the sample by defining the N value. samples contain index values for 
# #   the parameter lists. As the calculation for second order is defined as false, 
# #   the following equation will be used to defined the sample size: N∗(D+2) .
# #    As the values defined by Sobol and Morris are floats, they need to be rounded 
# #   and converted to int in order to be used as indices

# N = 2
# morris_sample_st = morris.sample(problem_st, N, num_levels=4, optimal_trajectories=None)
# morris_sample_st = morris_sample_st.round().astype(int)

# morris_sample_pvt = morris.sample(problem_pvt, N, num_levels=4, optimal_trajectories=None)
# morris_sample_pvt = morris_sample_st.round().astype(int)

# #%%% Creating and exporting sample csv

# st_out = assign_indices(morris_sample_st, input_st, ['design_case'], ['ST'])
# pvt_out = assign_indices(morris_sample_pvt, input_pvt, ['design_case'], ['PVT'])

# os.chdir(directory+sample_folder)
# # st_out.to_csv(output_file1+'.csv', index=False)
# # pvt_out.to_csv(output_file2+'.csv', index=False)

# #%%% fill existing results

# results = pd.read_csv('sim_results.csv', index_col='label')
# results['total_costs'] = results['el_bill']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000

# existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
# dfresults = pd.concat([existing, results],axis=1)

# st1 = pd.merge(st_out,dfresults, how='left')
# pvt1 = pd.merge(pvt_out,dfresults, how='left')

# #%%% flag missing data
# missing_st = st1[np.isnan(st1['el_bill'])]
# missing_pvt = pvt1[np.isnan(pvt1['el_bill'])]

# #%% dropping design case column since it did not exist in problem defintion
# #   Representing r_level in with corresponding digit
# #   Converting to numpy array of floats because that is the required input for morris.analyze()

# analyse_sample = st_out.drop('design_case', axis=1)
# analyse_sample['r_level'] = (analyse_sample['r_level'].str.extract('(\d+)'))
# analyse_sample = analyse_sample.to_numpy(dtype=float)
# #%%%
# from SALib.analyze import morris
# Si = morris.analyze(problem_st,
#                     analyse_sample,
#                     np.ravel(st1['el_bill']),
#                     conf_level=0.95,
#                     print_to_console=True,
#                     num_levels=4,
#                     num_resamples=100,)

# #%%% plots from SALib git hub

# from SALib.plotting.morris import (
#     horizontal_bar_plot,
#     covariance_plot,
#     sample_histograms,
# )
# fig, (ax1, ax2) = plt.subplots(1, 2)
# horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"EUR/year")
# covariance_plot(ax2, Si, {}, unit=r"EUR/year")

# fig2 = plt.figure()
# sample_histograms(fig2, analyse_sample, problem_st, {"color": "y"})
# plt.show()


# #%%% Sobol SA
# from SALib.sample import sobol
# N=2
# param_values = sobol.sample(problem_st, N, calc_second_order=True)
# sobol_sample = param_values.round().astype(int)
# df_out = assign_indices(sobol_sample, input_st, ['design_case'], ['ST'])

# df = pd.merge(df_out,dfresults, how='left')
# missing = df[np.isnan(df['el_bill'])]
# analyse_sample = df_out.drop('design_case', axis=1)
# analyse_sample['r_level'] = (analyse_sample['r_level'].str.extract('(\d+)'))

# #%%%
# from SALib.analyze import sobol
# analyse_sample = analyse_sample.to_numpy(dtype=float)
# Si = sobol.analyze(problem_st, 
#                    np.ravel(df['el_bill']), 
#                    calc_second_order=True, 
#                    conf_level=0.95, 
#                    print_to_console=True)


#%%% creating new samples using funtions
# problem, samp = prepare_sa('Sobol', input_st, ['design_case', 'r_level'], ['ST', 'r0'], N=4)

#%% running SA
# Si, missing = perform_sa('Sobol', 'el_bill', problem, samp, dfresults, ['design_case','r_level'])