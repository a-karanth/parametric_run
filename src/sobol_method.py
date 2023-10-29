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
input_st = {'volume' : [0.15, 0.2, 0.25],
              'coll_area': [4, 8, 16,20]}
              # 'flow_rate': [50, 100, 200]} #,
             # 'r_level': ['r0','r1']}

input_pvt = {'volume' : [0.15, 0.2, 0.25],
              'coll_area': [4, 8, 16,20]}
             # 'flow_rate': [50, 100, 200]}
           #   'r_level': ['r0','r1']}

input_cp = {'volume' : [0.15, 0.2, 0.25],
              'coll_area': [0.001, 4, 8, 16,20],
              'r_level': ['r0','r1']}


input_pvt_batt = {'volume' : [0.15, 0.2, 0.25],
              'coll_area': [4,8],
              'flow_rate': [50, 100, 200],
              'batt': [6, 9]}
              #'r_level': ['r0','r1']}


input_gen = {'volume' : [0.15, 0.2, 0.25],
             'coll_area': [4, 8, 16,20],
             'flow_factor': [25, 30, 35, 40],
             'design_case':['ST','PVT_0','PVT_6','PVT_9'],
             'r_level': ['r0','r1','r2']}

input_cp_ashp = {'volume' : [0.15, 0.2, 0.25],
             'coll_area': [0.001,4, 8, 16, 20], # if you create samples for ashp and cp at the 
             'design_case':['cp_PV','ASHP'],    # same time, you have to delete samples of ASHP
             'r_level': ['r0','r1','r2']}       # that have coll area 0.001

input_gen2 = {'volume' : [0.15, 0.2, 0.25],
             'coll_area': [4, 8, 16,20],
             'flow_rate': [50, 100, 200],
             'design_case':['ST','PVT_0','PVT_6','PVT_9','cp_PV'],
             'r_level': ['r0','r1']}

input_ashp = {'volume': [0.15, 0.2, 0.25],
              'coll_area': [4,8,16,20],
              'r_level': ['r0','r1']}

input_gen3 = {'volume' : [0.15, 0.2, 0.25],
             'coll_area': [4, 8, 16, 20],
             'flow_factor': [25, 30, 35, 40],
             'design_case':['ST','PVT_0','PVT_6','PVT_9'],
             'r_level': ['r0','r1']}

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
        bounds.append([0, len(dct[i])-1])
        nscenarios = nscenarios*len(dct[i])
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
    if key_name:
        for i,j in zip(key_name,values):
            result[i] = j
            if isinstance(j,str) and 'PVT_' in j: # if design_case contains PVT_batt
                j = 'PVT_' + result['batt'].astype(str)
                result[i] = j
            else:
                True
    result['flow_rate'] = result['coll_area']*result['flow_factor']
    result['inf'] = result['r_level'].apply(lambda x: 1 if x == 'r0' 
                                            else (0.4 if x == 'r1' else 0.2))
            
    return result

#%% function for preparing SA variables
from SALib.sample import sobol, morris

def prepare_sa(sa_type, ip, key_names=None, values=None, N=2, prnt=False, prefix='x'):
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
    #   As the values defined by Sobol and Morris are floats, they need to be rounded 
    #   and converted to int in order to be used as indices
    match sa_type:
        case 'sobol':
            samp = sobol.sample(problem, N, calc_second_order=True)
        case 'morris':
            samp = morris.sample(problem, N, num_levels=4, optimal_trajectories=None)
        
    samp = samp.round().astype(int)
    samp = assign_indices(samp, ip, key_names, values)
    if prnt:
        samp.to_csv('res\\'+sa_type+'_sample_'+ prefix +'.csv', index=False)
    return problem, samp

#%% LHS method
from SALib.sample import latin
ip = input_cp_ashp
bounds, nscenarios = cal_bounds_scenarios(ip)
problem = {'num_vars': len(ip),
           'names':list(ip.keys()),
           'bounds':bounds}
lhs_sample = latin.sample(problem,N=5000)
lhs_sample = lhs_sample.round().astype(int)
# lhs_sample = assign_indices(lhs_sample, ip, None, None)
lhs_sample = assign_indices(lhs_sample, ip, key_name=['flow_factor'], values=[0])

unique = lhs_sample.drop_duplicates(ignore_index=True)
perc = len(unique)/nscenarios
print(perc)
# unique.to_csv('res\\cp_sample_1.csv', index=False)

#%% remove values from unique such that design_case==ASHP and coll_area==0.001
remove = unique.query("design_case == 'ASHP' and coll_area == 0.001")
unique = unique.drop(remove.index)
#%% creating new samples
# problem, samp = prepare_sa('morris', input_cp, 
#                             ['design_case', 'flow_rate','r_level'], ['cp_PV', 100,'r0'], 
#                             N=16, prefix='cp3', prnt=True)
# problem2, samp2 = prepare_sa('morris', input_pvt, 
#                             ['design_case', 'flow_rate','r_level'], ['PVT', 100,'r0'], 
#                             N=16, prefix='pvt4', prnt=True)
# problem3, samp3 = prepare_sa('morris', input_st, 
#                             ['design_case', 'flow_rate','r_level'], ['ST', 100,'r0'], 
#                             N=16, prefix='st4', prnt=True)

#%% full factorial
from itertools import product
def full_factorial(inp, keys=None, values=None):
    combinations = list(product(*inp.values()))
    df = pd.DataFrame(combinations, columns=inp.keys())
    if keys:
        for i,j in zip(keys,values):
            df[i] = j
            df[i] = j
    df['flow_rate'] = df['coll_area']*df['flow_factor']
    df['inf'] = df['r_level'].apply(lambda x: 1 if x == 'r0' 
                                            else (0.4 if x == 'r1' else 0.2))
    return df

#%% example of creating a full factorial of all combination and then removing 
#   redundant combinations like:
#   All flow factors other and 0 for cp_PV and ASHP,
#   Area=0 for ST, ASHP, PVT_0, PV_6, PVT_9
# 
input_gen = {'volume' : [0.15, 0.2, 0.25],
             'coll_area': [0.001, 4, 8, 16,20],
             'flow_factor': [0, 25, 30, 35, 40],
             'design_case':['cp_PV', 'ASHP', 'ST','PVT_0','PVT_6','PVT_9'],
             'r_level': ['r0','r1','r2']}

test = full_factorial(input_gen) 

remove1 = test.query("(design_case == 'ASHP' or design_case=='cp_PV') and (flow_factor == 25 or flow_factor == 30 or flow_factor == 35 or flow_factor == 40)")
remove2 = test.query("(coll_area==0.001 or flow_factor==0) and (design_case=='ST' or design_case=='PVT_0' or design_case=='PVT_6' or design_case=='PVT_9')")
remove3 = test.query("design_case == 'ASHP' and coll_area ==0.001")
indices_to_remove = pd.concat([remove1, remove2, remove3])

test = test.drop(indices_to_remove.index)
# test.to_csv('res/ff_sample_1.csv',index=False)

#%% function to perform SA on the generated samples
from SALib.analyze import sobol as sobol_ana
from SALib.analyze import morris as morris_ana

def perform_sa(sa_type, kpi, problem, sample, sim_results, columns2drop, to_number=None):
    # to_number: which column has to be converted from string to number
    df = pd.merge(sample, sim_results, how='left')
    missing = df[np.isnan(df[kpi])]
    X = sample.drop(columns2drop, axis=1)
    if to_number:
        X[to_number] = (X[to_number].str.extract('(\d+)'))
    X = X.to_numpy(dtype=float)
    Y = df[kpi].ravel()             #to flatten series into a numpy array
    if len(missing != 0):
        return False, missing
    else:
        match sa_type:
            case 'sobol':
                Si = sobol_ana.analyze(problem, 
                                   Y, 
                                   calc_second_order=True, 
                                   conf_level=0.95, 
                                   print_to_console=True)
            case 'morris':
                Si = morris_ana.analyze(problem,
                                    X,
                                    Y,
                                    conf_level=0.95,
                                    print_to_console=True,
                                    num_levels=4,
                                    num_resamples=100,)
    return Si, missing

#%% collecting results
# results = pd.read_csv(res_folder+'sim_results.csv', index_col='label')
# results['total_costs'] = results['el_bill_1']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000

# existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
# existing['coll_area'] = existing['coll_area'].astype(int)
# dfresults = pd.concat([existing, results],axis=1)
#%% add a column to calculate battery size
# dfresults.insert(4,'batt',None)
# dfresults['batt'] = dfresults['design_case'].str.extract(r'(\d+)')
# dfresults['batt'] = dfresults['batt'].fillna(0).astype(int)
# dfresults = dfresults.drop_duplicates(ignore_index=True)

#%% Running a loop to find samples with all existing data
# count = 0
# sa_type = 'morris'
# while True:
#     print(count)
#     problem, samp = prepare_sa(sa_type, input_pvt_batt, ['r_level','design_case'], ['r0','PVT_Batt'], N=2)
#     Si, missing = perform_sa(sa_type, 'el_bill_0', problem, samp, dfresults, ['r_level','design_case'])
#     count = count + 1
#     if len(missing) == 0:
#         Si.plot()
#         break
# plt.gcf().set_size_inches(6,7)
# # plt.ylim([-95,120])
# plt.grid('both', linestyle='--')
# plt.tight_layout()
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