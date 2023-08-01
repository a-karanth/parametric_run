# -*- coding: utf-8 -*-
"""
use results from csv to analyze them

@author: 20181270
"""

import time                 # to measure the computation time
import os 
import multiprocessing as mp
# from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots as pt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

res_folder = 'res\\'
trn_folder = 'res\\trn\\'
#%% Reading resuls and calculating fianl kpis for comparison, assigning results 
#   based on samples 
results = pd.read_csv(res_folder+'sim_results.csv', index_col='label')
results['total_costs'] = results['el_bill']+results['gas_bill']
results['total_emission'] = (results['el_em']+results['gas_em'])/1000
existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')

dfresults = pd.concat([existing, results],axis=1)

dfmorris_st = pd.read_csv('res\\morris_st_sample2.csv')
morris_out_st = pd.merge(dfmorris_st, dfresults, on = ['volume','coll_area','flow_rate','design_case','r_level'], how = 'left')


dfmorris_pvt = pd.read_csv('res\\morris_pvt_sample2.csv')
morris_out_pvt = pd.merge(dfmorris_pvt, dfresults, on = ['volume','coll_area','flow_rate','design_case','r_level'], how = 'left')


# sample_st= dfmorris_st.copy()
# sample_st.drop(columns=['design_case'], inplace=True)
# sample_pvt= dfmorris_pvt.copy()
# sample_pvt.drop(columns=['design_case'], inplace=True)

#%% recreating problem - copied from sobol_method
# input_st = {'volume' : [0.1, 0.2, 0.3, 0.4],
#               'coll_area': [4, 8, 16,20],
#               'flow_rate': [50, 100, 200],
#               'r_level': ['r0','r1']}

# input_pvt = {'volume' : [0.1, 0.2, 0.3, 0.4],
#               'coll_area': [4, 8, 16,20],
#               'flow_rate': [50, 100, 200],
#               'r_level': ['r0','r1']}

# def cal_bounds_scenarios(dct):
#     key = list(dct.keys())
#     bounds = []
#     nscenarios = 1
#     for i in key:
#         bounds.append([0, len(input_st[i])-1])
#         nscenarios = nscenarios*len(input_st[i])
    # return bounds, nscenarios

#%% Creating bounds and scenarios
# bounds_st, nscenarios_st = cal_bounds_scenarios(input_st)
# bounds_pvt, nscenarios_pvt = cal_bounds_scenarios(input_pvt)

#%% Creating problems
# problem_st = {
#     'num_vars': len(input_st),
#     'names':list(input_st.keys()),
#     'bounds':bounds_st}

# problem_pvt = {
#     'num_vars': len(input_pvt),
#     'names':list(input_pvt.keys()),
#     'bounds':bounds_pvt}

#%% analysing morris samples
# Si = morris.analyze(
#     problem_pvt,
#     sample_pvt,
#     np.ravel(morris_out_pvt['total_costs']),
#     conf_level=0.95,
#     print_to_console=True,
#     num_levels=4,
#     num_resamples=100,
# )

#%% Focussing on one volume

# morris_out_pvt['volume'] = morris_out_pvt['volume']-0.004
# morris_out_st['volume'] = morris_out_st['volume']+0.004
fil_pvt = morris_out_pvt[morris_out_pvt['volume']==0.2]
fil_st = morris_out_st[morris_out_st['volume']==0.2]

#%% Scatter plots

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize = (19,9))   
scatter1, cbar1 = pt.scatter_plot(morris_out_pvt, morris_out_st, ax=ax1,
                       marker=['^', 'P'], 
                       xkey='volume', ykey='total_costs',ckey='coll_area',
                       xlabel='Volume [m3]', ylabel='Total cost [EUR]', clabel='Coll area [m2]')

scatter2, cbar2 = pt.scatter_plot(morris_out_pvt, morris_out_st, ax=ax2,
                       marker=['^', 'P'], 
                       xkey='volume', ykey='total_costs',ckey='flow_rate',
                       xlabel='Volume [m3]', ylabel='Total cost [EUR]', clabel='Flow rate [kg/hr]')

scatter3, cbar3 = pt.scatter_plot(fil_pvt, ax=ax3,
                       marker=['^', 'P'], 
                       xkey='flow_rate', ykey='total_costs',ckey='coll_area',
                       xlabel='Flow rate [kg/hr]', ylabel='Total cost [EUR]', clabel='Coll area [m2]')

#%%
# def plotly_plots():
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
pio.renderers.default = 'browser'

fil_pvt['label'] = fil_pvt.index
fil_pvt['r_number'] = fil_pvt['r_level']
fil_pvt['r_number']=fil_pvt['r_number'].replace('r0',0+1)
fil_pvt['r_number']=fil_pvt['r_number'].replace('r1',1+3)

fil_st['label'] = fil_st.index
fig = px.scatter(fil_pvt, x="flow_rate", y="total_costs", color='coll_area', size='r_number',  
                 hover_data=['label'], size_max=20)
fig.update_traces(marker=dict(symbol='square'))

st =  px.scatter(fil_st, x="flow_rate", y="total_costs", color='coll_area',  
                 hover_data=['label'], size_max=20)
st.update_traces(marker=dict(symbol='triangle-up', size=20))
fig.add_trace(st.data[0])
fig.show()

#%%
# pd.options.plotting.backend = "plotly"
fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.05, 
                    row_heights=[0.5,0.05],
                    specs=[[{"secondary_y":True}],[{"secondary_y":True}],
                            [{"secondary_y":True}],[{"secondary_y":True}],[{"secondary_y":False}]],
                    subplot_titles=("flow rate [kg/hr]", "coll area [m2]"))

temperatures = ['Thp_load_out','Tsh_in','Tdhw_in', 'Thp_load_in']
mass_flow = ['mhp_load_out','msh_in', 'mdhw_in', 'mhp_load_in']

fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tamb'], name='Tamb'), secondary_y=False,row=1, col=1)
fig.add_trace(go.Scatter(x=energy.index, y=energy['Qirr'], name='Qirr'), secondary_y=True,row=1, col=1)
fig.update_layout(yaxis=dict(title="temperature [deg C]"), yaxis2=dict(title="irradiance [kW]"))

for i in temperatures:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=2, col=1)
for i in mass_flow:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=2, col=1)
fig.update_layout(yaxis3=dict(title="temperature [deg C]"), yaxis4=dict(title="flow rate [kg/hr]"))

temperatures = ['T1_sh','Tavg_sh','T6_sh', 'Tsh_cold_out', 'Trad1_in','Tsh_return','Tfloor1','Tfloor2']
mass_flow = ['msh_cold_out','mrad1_in','mrad2_in', 'msh_return']
for i in temperatures:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=3, col=1)
for i in mass_flow:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=3, col=1)
fig.update_layout(yaxis5=dict(title="temperature [deg C]"), yaxis6=dict(title="flow rate [kg/hr]"))

temperatures = ['T1_dhw','Tavg_dhw','T6_dhw', 'Tdhw_cold_out', 'Tdhw2tap','Tat_tap']
mass_flow = ['mdhw_cold_out','mdhw2tap','mcold2tap', 'mat_tap']
for i in temperatures:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=4, col=1)
for i in mass_flow:
    fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=4, col=1)
fig.update_layout(yaxis7=dict(title="temperature [deg C]"), yaxis8=dict(title="flow rate [kg/hr]"))

cntr = ['hp_pump','hp_div','ctr_sh','ctr_dhw','heatingctr1','heatingctr2']
for i in cntr:
    fig.append_trace(go.Scatter(x=temp_flow.index, y=controls[i], name=i),row=5, col=1)
fig.update_layout(title_text='Output files: ' + output_prefix)
fig.update_layout(xaxis_range=[t1,t2])
fig.show()
#temp_flow.plot(y=['Thp_load_out','Tsh_in','Tdhw_in', 'Thp_load_in'])
#offline.plot(fig,filename='temp.html')
#%%S
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