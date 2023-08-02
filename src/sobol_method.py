# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:49:51 2023

@author: 20181270
"""

# # Perform the sensitivity analysis using the model output
# # Specify which column of the output file to analyze (zero-indexed)
# Si = morris.analyze(
#     problem,
#     param_values,
#     Y,
#     conf_level=0.95,
#     print_to_console=True,
#     num_levels=4,
#     num_resamples=100,
# )
# # Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# # e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# # same order as the parameter file


import os
import sys
import pkgutil
from warnings import warn
from SALib.analyze import sobol, morris
from SALib.test_functions import Ishigami
from SALib import analyze, sample, plotting
from SALib.sample import latin, saltelli, morris
# from SALib.plotting import morris
# from SALib.plotting.bar import plot as barplot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from collections import OrderedDict


directory = (os.path.dirname(os.path.realpath(__file__)))
sample_folder = '\\res'

output_file1 = 'morris_sample_st3'
output_file2 = 'morris_sample_pvt3'

#create one list for each variable indicating their input values
input_st = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200],
              'r_level': ['r0','r1']}

input_pvt = {'volume' : [0.1, 0.2, 0.3, 0.4],
              'coll_area': [4, 8, 16,20],
              'flow_rate': [50, 100, 200],
              'r_level': ['r0','r1']}


list_design_case_pvt_batt = ['PVT_Batt_6', 'PVT_Batt_9']

#%% Function for creating bounds and scenarios
#   define problem to be analysed (number, name and range of the variables)
#   creating the problem based on list index numbers. Which will then be used to 
#   create final sample by applying to the lists declared above

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


#%% Create the sample by defining the N value. samples contain index values for 
#   the parameter lists. As the calculation for second order is defined as false, 
#   the following equation will be used to defined the sample size: N∗(D+2) .
#    As the values defined by Sobol and Morris are floats, they need to be rounded 
#   and converted to int in order to be used as indices

# param_values = saltelli.sample(problem, N, calc_second_order=True)
# sobol_sample = param_values.round()

N = 2
morris_sample_st = morris.sample(problem_st, N, num_levels=4, optimal_trajectories=None)
morris_sample_st = morris_sample_st.round().astype(int)

morris_sample_pvt = morris.sample(problem_pvt, N, num_levels=4, optimal_trajectories=None)
morris_sample_pvt = morris_sample_st.round().astype(int)


#%%
def assign_indices(samples, inp, design_case):
    # samples: sample created by morris or sobol classes
    # inp: dictionary if inputs containing the different parameters
    # design_case: since design case could not be added in the input, adding 
    # last column as the name of the design case
    keys = list(inp.keys())
    result = pd.DataFrame(columns=keys)
    for row in np.arange(len(samples)):
        result.loc[row] = samples[row]
        for i,k in zip(samples[row], keys):
            result[k].loc[row] = inp[k][i]
    result.columns = keys
    result['design_case'] = design_case
    return result

#%% Creating and exporting sample csv

st_out = assign_indices(morris_sample_st, input_st, 'ST')
pvt_out = assign_indices(morris_sample_pvt, input_pvt, 'PVT')

os.chdir(directory+sample_folder)
# st_out.to_csv(output_file1+'.csv', index=False)
# pvt_out.to_csv(output_file2+'.csv', index=False)


#%% fill existing results
trn_folder = 'trn\\'
results = pd.read_csv('sim_results.csv', index_col='label')
results['total_costs'] = results['el_bill']+results['gas_bill']
results['total_emission'] = (results['el_em']+results['gas_em'])/1000

existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
dfresults = pd.concat([existing, results],axis=1)

st1 = pd.merge(st_out,dfresults, how='left')
pvt1 = pd.merge(pvt_out,dfresults, how='left')

#%% flag missing data
missing_st = st1[np.isnan(st1['el_bill'])]
missing_pvt = pvt1[np.isnan(pvt1['el_bill'])]

#%% dropping design case column since it did not exist in problem defintion
#   Representing r_level in with corresponding digit
#   Converting to numpy array of floats because that is the required input for morris.analyze()

analyse_sample = st_out.drop('design_case', axis=1)
analyse_sample['r_level'] = (analyse_sample['r_level'].str.extract('(\d+)'))
analyse_sample = analyse_sample.to_numpy(dtype=float)
#%%
from SALib.analyze import morris
Si = morris.analyze(
    problem_st,
    analyse_sample,
    np.ravel(st1['el_bill']),
    conf_level=0.95,
    print_to_console=True,
    num_levels=4,
    num_resamples=100,
)

#%% plots from SALib git hub

from SALib.plotting.morris import (
    horizontal_bar_plot,
    covariance_plot,
    sample_histograms,
)
fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"EUR/year")
covariance_plot(ax2, Si, {}, unit=r"EUR/year")

fig2 = plt.figure()
sample_histograms(fig2, analyse_sample, problem_st, {"color": "y"})
plt.show()

#%%
# #Give names to the file containing all the simulation outputs (DS_results.csv) and 
# #the one containing the Sobol sample (SA_sample.csv)
# os.chdir(results_SAn_DA)
# dfsobol = pd.read_csv(results_SAn_DA+'\\SAn_DA_sample.csv')
# # dfresults = pd.read_csv(results_SAn_DA+'\\Simulation_results\\DA_revised.csv')
# print(len(dfsobol))
# # print(len(dfresults))

#Check how many unique scenarios the sobol analysis have 
# pd.options.display.multi_sparse = False
# unique_sample = (dfsobol.value_counts())

#Calculate the percentage of unique scenarios there is in the Sobol sample
# percentage = (round((len(unique_sample)*100/n_scenarios)))
# print(str(percentage) + '%')

# #Go to the file containing all the simulation outputs (DS_results.csv) and order them according to the Sobol sample (SA_out.csv)

# # Sobol_out = pd.merge(dfsobol, dfresults, on = ['Balcony Location','Balcony Depth',
# # 'Parapet','Gtvis', 'Window Width', 'Orientation'], how = 'left')
# Sobol_out = dfsobol
# print (len(Sobol_out))
# Sobol_out.to_csv('Sobol_out.csv', index=False)

# Si = sobol.analyze(problem, np.ravel(Sobol_out['Parapet']))
# Si.plot()

