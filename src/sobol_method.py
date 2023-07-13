# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:49:51 2023

@author: 20181270
"""

# import sys

# from SALib.analyze import morris
# from SALib.sample.morris import sample
# from SALib.test_functions import Sobol_G
# from SALib.util import read_param_file
# from SALib.plotting.morris import (
#     horizontal_bar_plot,
#     covariance_plot,
#     sample_histograms,
# )
# import matplotlib.pyplot as plt

# sys.path.append("../..")

# # Read the parameter range file and generate samples
# # problem = read_param_file("../../src/SALib/test_functions/params/Sobol_G.txt")
# # or define manually without a parameter file:
# problem = {
#   'num_vars': 8,
#   'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
#   'groups': None,
#   'bounds': [[0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0],
#             [0.0, 1.0]]
# }
# # Files with a 4th column for "group name" will be detected automatically, e.g.
# # param_file = '../../src/SALib/test_functions/params/Ishigami_groups.txt'

# # Generate samples
# param_values = sample(problem, N=1000, num_levels=4, optimal_trajectories=None)

# # To use optimized trajectories (brute force method),
# # give an integer value for optimal_trajectories

# # Run the "model" -- this will happen offline for external models
# Y = Sobol_G.evaluate(param_values)

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

# fig, (ax1, ax2) = plt.subplots(1, 2)
# horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"tCO$_2$/year")
# covariance_plot(ax2, Si, {}, unit=r"tCO$_2$/year")

# fig2 = plt.figure()
# sample_histograms(fig2, param_values, problem, {"color": "y"})
# plt.show()

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import os
import sys
from SALib import analyze, sample, plotting
from SALib.analyze import sobol, morris
from SALib.sample import latin, saltelli, morris
from SALib.plotting import morris
from SALib.plotting.bar import plot as barplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import csv
from warnings import warn
import pkgutil
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker

directory = (os.path.dirname(os.path.realpath(__file__)))
results_SAn_DA = directory

#create one list for each variable indicating their input values
list_volume = [0.1, 0.15, 0.2, 0.25, 0.3]
list_coll_area = [10, 12, 14, 16, 18, 20]
list_design_case = ['cp','cp_PV', 'ST', 'PVT', 'PVT_Batt_6', 'PVT_Batt_9']
# list_r = ['r0', 'r1', 'r2']

#define problem to be analysed (number, name and range of the variables)
# creating the problem based on list index numbers. Which will then be used to create final sample
# by applying to the lists declared above

problem = {
    'num_vars': 3,
    'names':['tank_volume', 'coll_rea', 'design_case'],#, 'r_values'],
    'bounds':[[0, len(list_volume)-1],
              [0, len(list_coll_area)-1],
              [0, len(list_design_case)-1],]}
              # [0, len(list_r)-1]]}

# Create the sample by defining the N value. As the calculation for second order 
# is defined as false, the following equation will be used to defined the sample size: N∗(D+2) 
N = 32
param_values = saltelli.sample(problem, N, calc_second_order=True)
print(param_values)

# As the values defined by Sobol are floats, they need to be rounded in order to 
# fit my case with discrete variables
sobol_sample = param_values.round()
    
n_scenarios = len(list_volume) * len(list_coll_area) * len(list_design_case) #* len(list_r)

#to create an excel file with the Sobol sample, first create empty lists with the outputs name
   
variable_out = pd.DataFrame()
variable_out['volume'] = [list_volume[int(sobol_sample[i][0])] for i in range(len(sobol_sample))]
variable_out['coll_area'] = [list_coll_area[int(sobol_sample[i][1])] for i in range(len(sobol_sample))]
variable_out['design_case'] = [list_design_case[int(sobol_sample[i][2])] for i in range(len(sobol_sample))]
# variable_out['R_values'] = [list_r[int(sobol_sample[i][3])] for i in range(len(sobol_sample))]

# #Create a csv file with the SA outputs
# os.chdir(results_SAn_DA)
variable_out.to_csv('SAn_DA_sample'+'.csv', index=False)

# #Give names to the file containing all the simulation outputs (DS_results.csv) and 
# #the one containing the Sobol sample (SA_sample.csv)
# os.chdir(results_SAn_DA)
dfsobol = pd.read_csv(results_SAn_DA+'\\SAn_DA_sample.csv')
# # dfresults = pd.read_csv(results_SAn_DA+'\\Simulation_results\\DA_revised.csv')
# print(len(dfsobol))
# # print(len(dfresults))

#Check how many unique scenarios the sobol analysis have 
pd.options.display.multi_sparse = False
unique_sample = (dfsobol.value_counts())

#Calculate the percentage of unique scenarios there is in the Sobol sample
percentage = (round((len(unique_sample)*100/n_scenarios)))
# print(str(percentage) + '%')

# #Go to the file containing all the simulation outputs (DS_results.csv) and order them according to the Sobol sample (SA_out.csv)

# # Sobol_out = pd.merge(dfsobol, dfresults, on = ['Balcony Location','Balcony Depth',
# # 'Parapet','Gtvis', 'Window Width', 'Orientation'], how = 'left')
# Sobol_out = dfsobol
# print (len(Sobol_out))
# Sobol_out.to_csv('Sobol_out.csv', index=False)

# Si = sobol.analyze(problem, np.ravel(Sobol_out['Parapet']))
# Si.plot()

