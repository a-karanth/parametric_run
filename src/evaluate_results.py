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
results['total_costs_1'] = results['el_bill_1']+results['gas_bill']
results['total_costs_0.5'] = results['el_bill_0.5']+results['gas_bill']
results['total_costs_0.1'] = results['el_bill_0.1']+results['gas_bill']
results['total_costs_0'] = results['el_bill_0']+results['gas_bill']
results['total_emission'] = (results['el_em']+results['gas_em'])/1000
existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')

dfresults = pd.concat([existing, results],axis=1)

#%% add a column to calculate battery size
dfresults.insert(4,'batt',None)
dfresults['batt'] = dfresults['design_case'].str.extract(r'(\d+)')
dfresults['batt'] = dfresults['batt'].fillna(0).astype(int)
df = dfresults.copy()
#%% Focussing on one volume
df = dfresults[(dfresults['r_level']=='r0') ]#& (dfresults['volume']==0.25)]
fil = df[df['volume']==0.25]

#%% Scatter plots

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize = (19,9))   
# scatter1, cbar1 = pt.scatter_plot(df[(df['design_case']=='ST') & (df['r_level']=='r1')], 
#                                   df[(df['design_case']=='PVT') & (df['r_level']=='r1')], 
#                                   ax=ax1, marker=['^', 'P'], 
#                                   xkey='volume', ykey='total_costs',ckey='coll_area',
#                                   xlabel='Volume [m3]', ylabel='Total cost [EUR]', 
#                                   clabel='Coll area [m2]')

scatter2, cbar2 = pt.scatter_plot(df[df['design_case']=='ST'], 
                                  df[df['design_case']=='PVT'], 
                                  ax=ax2, marker=['^', 'P'], 
                                  xkey='volume', ykey='total_costs',ckey='flow_rate',
                                  xlabel='Volume [m3]', ylabel='Total cost [EUR]', 
                                  clabel='Flow rate [kg/hr]')

scatter3, cbar3 = pt.scatter_plot(df[df['design_case']=='ST'],
                                  df[df['design_case']=='PVT'],
                                  df[df['design_case']=='PVT_Batt_6'],
                                  df[df['design_case']=='PVT_Batt_9'],
                                  df[df['design_case']=='cp_PV'], ax=ax3,
                                  marker=['^', 'o', 's','D'], 
                                  xkey='total_costs', ykey='total_emission',ckey='coll_area',
                                  xlabel='Total cost [EUR]', ylabel='Total emission [kgCO2]', 
                                  clabel='Coll area [m2]')

#%% Plot electricity bill for diff levels of net metering
fig, ax = plt.subplots(figsize=(6,7))
st = df[df['design_case']=='ST']
pvt = df[df['design_case']=='PVT']
batt6 = df[df['design_case']=='PVT_Batt_6']
batt9 = df[df['design_case']=='PVT_Batt_9']
cp = df[df['design_case']=='cp_PV']
marker_size= 50

df_plot = {'df':[st,pvt,batt6, batt9,cp],
        'marker':['^','o','o','o','s'],
        'color':['red','purple','orange','green','black'],
        'alpha':[1,1,1,1,0.5],
        'size':[marker_size,marker_size,marker_size,marker_size,50]}
x_values = [1, 2, 3, 4]
for i in x_values:
    match i:
        case 1:
            kpi = 'el_bill_1'
        case 2:
            kpi = 'el_bill_0.5'
        case 3:
            kpi = 'el_bill_0.1'
        case 4:
            kpi = 'el_bill_0'
    for data in range(len(df_plot)):
        total_cost = df_plot['df'][data][kpi] + df_plot['df'][data]['gas_bill']
        ax.scatter([i]*len(df_plot['df'][data]),
                   total_cost,
                   marker = df_plot['marker'][data],
                   c='white',
                   edgecolors =df_plot['color'][data],
                   s =df_plot['df'][data]['flow_rate'],
                   alpha =df_plot['alpha'][data])
                   # label = df_plot['df'][data]['design_case'].iloc[0])

ax.legend(['ST', 'PVT','PVT_Batt_6','PVT_Batt_9','cp_PV'],loc='best')
ax.set_xlabel('% net metering')
ax.set_ylabel('Energy bill')
ax.set_title('Volume = 250 L')
ax.set_ylim(1200,4300)
plt.xticks(x_values, ['1','0.5','0.1','0'])
# ax.legend()

#%% plot emissions
df = dfresults[(dfresults['r_level']=='r0') ]
st = df[df['design_case']=='ST']
pvt = df[df['design_case']=='PVT']
batt6 = df[df['design_case']=='PVT_Batt_6']
batt9 = df[df['design_case']=='PVT_Batt_9']
cp = df[df['design_case']=='cp_PV']
fig, ax = plt.subplots()
ax.scatter(st['coll_area'],st['total_emission'],c='white',edgecolor='red',marker='^',label='ST')
ax.scatter(pvt['coll_area'],pvt['total_emission'],c='white',edgecolor='purple',marker='o',label='PVT')
ax.scatter(batt6['coll_area'],batt6['total_emission'],c='white',edgecolor='orange',marker='o',label='PVT_batt_6')
ax.scatter(batt9['coll_area'],batt9['total_emission'],c='white',edgecolor='green',marker='o',label='PVT_Batt_9')
ax.scatter(cp['coll_area'],cp['total_emission'],c='white',edgecolor='black',marker='s',label='cp_PV')

ax.set_xlabel('collector area [m2]')
ax.set_ylabel('Emissions kgCO2/year')
ax.set_title('Carbon emissions')

best_cp = cp[cp.el_bill_1==cp.el_bill_1.min()]
best_st =st[st.el_bill_1==st.el_bill_1.min()] 
best_pvt = pvt[pvt.el_bill_1==pvt.el_bill_1.min()]
best_batt6 = batt6[batt6.el_bill_1==batt6.el_bill_1.min()]
best_batt9 = batt9[batt9.el_bill_1==batt9.el_bill_1.min()]

#%% parallel coordinate plot
from pandas.plotting import parallel_coordinates

df = dfresults.copy()
df['design_case'] = df['design_case'].replace(['cp_PV','ST','PVT_0','PVT_6','PVT_9'],
                                              [0,1,2,3,4])
df['r_level'] = df['r_level'].replace(['r0','r1'],[0,1])

parallel_coordinates(df[['volume','coll_area','flow_rate','r_level','design_case','el_bill_1']],
                     'coll_area', colormap=plt.get_cmap("viridis"))

#%% plotly express
import plotly.express as px
fig = px.parallel_coordinates(df, color="r_level", 
                              labels={"coll_area": "Coll_area","volume": "Volume",
                                      "flow_rate": "Flow rate","design_case": "Design Case",
                                      "batt": "Batt size", },
                              dimensions=['volume', 'coll_area', 'flow_rate','design_case',
                                          'batt','r_level','el_bill_1','el_bill_0.5','el_bill_0.1',
                                          'el_bill_0','total_emission'],
                              color_continuous_scale=px.colors.sequential.Viridis,
                              color_continuous_midpoint=0.5)
fig.show()

#%% plotly graph objects
import plotly.graph_objects as go

fig = go.Figure(data=
                go.Parcoords(
                    line = dict(color = df['coll_area'],
                                colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']],
                                # colorscare = 'Electric',
                                showscale=True,
                                colorbar=dict(title='Coll area [m2]')),  # Add the colorbar title here,
                    dimensions = list([dict(tickvals = [0,1,2,3,4],
                                            label = 'Design case', values = df['design_case'],
                                            ticktext = ['cp_PV', 'ST', 'PVT', 'PVT  6','PVT 9']),
                                       dict(label = 'R level', values = df['r_level']),
                                       dict(#range = [1,5],
                                            #constraintrange = [1,2], # change this range by dragging the pink line
                                            label = 'Volume', values = df['volume']),        
                                       dict(label = 'Coll area', values = df['coll_area']),
                                       dict(label = 'Flow rate', values = df['flow_rate']),
                                       dict(label = 'Q4sh', values = df['Q4sh']),
                                            # range = [1000,3500]),
                                       dict(label = 'Q4dhw', values = df['Q4dhw']),
                                            # range = [0,1000]),
                                       dict(label = 'Qaux', values = df['Qaux']),
                                       dict(label = 'El bill 0', values = df['el_bill_0']),
                                       dict(label = 'El bill 1', values = df['el_bill_1']),
                                       dict(label = 'Total cost 0', values = df['total_costs_0']),
                                       dict(label = 'Total cost 1', values = df['total_costs_1'])
                                       ])
                    )
                )
fig.show()
#%% tutorial on plotly.com
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['species_id'],
                   colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']]),
        dimensions = list([
            dict(range = [0,8],
                constraintrange = [4,8],
                label = 'Sepal Length', values = df['sepal_length']),
            dict(range = [0,8],
                label = 'Sepal Width', values = df['sepal_width']),
            dict(range = [0,8],
                label = 'Petal Length', values = df['petal_length']),
            dict(range = [0,8],
                label = 'Petal Width', values = df['petal_width'])
        ])
    )
)
#%%
fig, axx = plt.subplots()
# df = dfresults.copy()
scatter3, cbar3 = pt.scatter_plot(df[df['design_case']=='ST'],
                                  df[df['design_case']=='PVT'],
                                  df[df['design_case']=='PVT_Batt_6'],
                                  df[df['design_case']=='PVT_Batt_9'],
                                  ax=axx,
                                  marker=['^', 'o', 's','D'], 
                                  xkey='flow_rate', ykey='el_bill_0',ckey='coll_area',
                                  xlabel='flow_rate', ylabel='Total cost [EUR]', 
                                  clabel='Coll area [m2]')
axx.set_ylim([1000,2600])
#%% Plotlt plots
# def plotly_plots():
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
pio.renderers.default = 'browser'
fil_pvt = dfresults[(dfresults['design_case']=='cp_PV') & (dfresults['r_level']=='r0') &
                    (dfresults['volume']==0.2)]

fil_pvt['label2'] = fil_pvt.index
# fil_pvt['r_number'] = fil_pvt['r_level']
# fil_pvt['r_number']=fil_pvt['r_number'].replace('r0',0+1)
# fil_pvt['r_number']=fil_pvt['r_number'].replace('r1',1+3)

# fil_st['label'] = fil_st.index
fig = px.scatter(fil_pvt, x="volume", y="total_costs", color='coll_area', size='coll_area',  
                 hover_data=['label2'], size_max=20)
fig.update_traces(marker=dict(symbol='square'))

# st =  px.scatter(fil_st, x="flow_rate", y="total_costs", color='coll_area',  
#                  hover_data=['label'], size_max=20)
# st.update_traces(marker=dict(symbol='triangle-up', size=20))
# fig.add_trace(st.data[0])
fig.show()

#%%
# pd.options.plotting.backend = "plotly"
# fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.05, 
#                     row_heights=[0.5,0.05],
#                     specs=[[{"secondary_y":True}],[{"secondary_y":True}],
#                             [{"secondary_y":True}],[{"secondary_y":True}],[{"secondary_y":False}]],
#                     subplot_titles=("flow rate [kg/hr]", "coll area [m2]"))

# temperatures = ['Thp_load_out','Tsh_in','Tdhw_in', 'Thp_load_in']
# mass_flow = ['mhp_load_out','msh_in', 'mdhw_in', 'mhp_load_in']

# fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow['Tamb'], name='Tamb'), secondary_y=False,row=1, col=1)
# fig.add_trace(go.Scatter(x=energy.index, y=energy['Qirr'], name='Qirr'), secondary_y=True,row=1, col=1)
# fig.update_layout(yaxis=dict(title="temperature [deg C]"), yaxis2=dict(title="irradiance [kW]"))

# for i in temperatures:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=2, col=1)
# for i in mass_flow:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=2, col=1)
# fig.update_layout(yaxis3=dict(title="temperature [deg C]"), yaxis4=dict(title="flow rate [kg/hr]"))

# temperatures = ['T1_sh','Tavg_sh','T6_sh', 'Tsh_cold_out', 'Trad1_in','Tsh_return','Tfloor1','Tfloor2']
# mass_flow = ['msh_cold_out','mrad1_in','mrad2_in', 'msh_return']
# for i in temperatures:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=3, col=1)
# for i in mass_flow:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=3, col=1)
# fig.update_layout(yaxis5=dict(title="temperature [deg C]"), yaxis6=dict(title="flow rate [kg/hr]"))

# temperatures = ['T1_dhw','Tavg_dhw','T6_dhw', 'Tdhw_cold_out', 'Tdhw2tap','Tat_tap']
# mass_flow = ['mdhw_cold_out','mdhw2tap','mcold2tap', 'mat_tap']
# for i in temperatures:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=False, row=4, col=1)
# for i in mass_flow:
#     fig.add_trace(go.Scatter(x=temp_flow.index, y=temp_flow[i], name=i), secondary_y=True, row=4, col=1)
# fig.update_layout(yaxis7=dict(title="temperature [deg C]"), yaxis8=dict(title="flow rate [kg/hr]"))

# cntr = ['hp_pump','hp_div','ctr_sh','ctr_dhw','heatingctr1','heatingctr2']
# for i in cntr:
#     fig.append_trace(go.Scatter(x=temp_flow.index, y=controls[i], name=i),row=5, col=1)
# fig.update_layout(title_text='Output files: ' + output_prefix)
# fig.update_layout(xaxis_range=[t1,t2])
# fig.show()
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

#%% check which is missing or repeated
list_label = np.arange(264)
check = pd.DataFrame(index=list_label)
check['count'] = 0
for i in existing.index:
    if i in list_label:
        check.loc[i] = check.loc[i]+1