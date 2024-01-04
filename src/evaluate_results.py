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
results['total_emission'] = results['el_em']+results['gas_em']
existing = pd.read_csv(trn_folder+'list_of_inputs.csv',header=0, index_col='label').sort_values(by='label')
results['total_cost_jan_0'] =  results['el_bill_jan_0']+results['gas_bill_jan']
dfresults = pd.concat([existing, results],axis=1)
rldc = pd.read_csv(res_folder+'rldc.csv',index_col=0)

#%% add a column to calculate battery size
dfresults.insert(4,'batt',None)
dfresults['batt'] = dfresults['design_case'].str.extract(r'(\d+)')
dfresults['batt'] = dfresults['batt'].fillna(0).astype(int)
df = dfresults.copy()

#%% convert string categories into numerical
df['design_case'] = df['design_case'].replace(['cp_PV','ST','ASHP','PVT_0','PVT_6','PVT_9'],
                                              [0,1,2,3,4,5])
df['r_level'] = df['r_level'].replace(['r0','r1','r2'],[0,1,2])

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

#%% costs vs investment costs
# df = dfresults.copy()
# fig, ax= plt.subplots()
# ax.scatter(df['cost'], df['total_costs_0'])

import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
import plotly.express as px

# df = df.sort_values(by='design_case')
# df['design_case'] = df['design_case'].replace([0,1,2,3,4,5],
#                                               ['cp_PV','ST','ASHP','PVT_0','PVT_6','PVT_9'])
fig = px.scatter(df, 
                 y='total_costs_0', x="cost", 
                 symbol="r_level", 
                 color="design_case",
                 symbol_sequence= [0,1,2,3,4,5],
                 # facet_col="time",
                 labels={"r_level": "r_level", "design_case": "Design Case",'coll_area':'Coll area'},
                 color_discrete_map={"cp_PV": "grey",
                                      "ST": "red",
                                      'ASHP':'Purple',
                                      "PVT_0": "limegreen",
                                      "PVT_6": "teal",
                                      "PVT_9": "darkblue"},
                 # color_discrete_sequence=px.colors.qualitative.Bold,
                 # color_continuous_scale="oxy",
                  title="Initial investvent cost vs Annual operational costs")
fig.update_traces(marker=dict(size=10))
fig.update_xaxes(range=[0, 20000])  # Adjust the range as needed for the x-axis
# fig.update_yaxes(range=[1.9, 3]) 
fig.update_layout(legend=dict(x=0, y=0))
fig.update_layout(legend=dict(yanchor="bottom", y=0.01,
                              xanchor="right", x=0.5))
# fig.update_layout(legend=dict(orientation="h", 
#                               yanchor="bottom", y=1.02, 
#                               xanchor="right", x=1),
#                   grid=dict(rows=3, columns=4),
#                   margin=dict(t=50, l=0, r=0, b=0),
#                   # showlegend=False  # Hide default legend
#                   )
fig.show()

#%% plotly PCP 1 - cost, emission, penalty
import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = 'browser'
fig = go.Figure(data=
                go.Parcoords(
                    line = dict(color = df['coll_area'],
                                colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']],
                                # colorscare = 'Electric',
                                showscale=True,
                                colorbar=dict(title='Coll area [m2]')),  # Add the colorbar title here,
                    dimensions = list([dict(label = 'R level', values = df['r_level']),
                                       dict(tickvals = [0,1,2,3,4,5],
                                            label = 'Design case', values = df['design_case'],
                                            ticktext = ['cp_PV', 'ST','ASHP', 'PVT', 'PVT  6','PVT 9']),
                                       dict(#range = [1,5],
                                            #constraintrange = [1,2], # change this range by dragging the pink line
                                            label = 'Volume', values = df['volume']),        
                                       # dict(label = 'Coll area', values = df['coll_area']),
                                        dict(label = 'Flow rate', values = df['flow_rate']),
                                       dict(label = 'Total cost 0', values = df['total_costs_0']),
                                       dict(label = 'Total emissions', values = df['total_emission']),
                                       # dict(label = 'Penalty', values = df['penalty_in'])
                                       ])
                    )
                )
fig.show()

#%% PCP 2: peak load, export, cost, penalty
fig = go.Figure(data=
                go.Parcoords(
                    line = dict(color = df['coll_area'],
                                # colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']],
                                colorscale = [[0,'orange'],[0.8,'lightseagreen'],[1,'purple']],
                                showscale=True,
                                colorbar=dict(title='Coll area [m2]')),  # Add the colorbar title here,
                    dimensions = list([dict(tickvals = [0,1,2,3,4,5],
                                            label = 'Design case', values = df['design_case'],
                                            ticktext = ['cp_PV', 'ST','ASHP', 'PVT', 'PVT  6','PVT 9']),
                                       dict(label = 'R level', values = df['r_level']),
                                       dict(#range = [1,5],
                                            #constraintrange = [1,2], # change this range by dragging the pink line
                                            label = 'Volume', values = df['volume']),        
                                       dict(label = 'Coll area', values = df['coll_area']),
                                       dict(label = 'Flow rate [l/h]', values = df['flow_rate']),
                                       dict(label = 'Peak load [kW]', values = df['peak_load']),
                                       dict(label = 'Peak export [kW]', values = df['peak_export']),
                                       dict(label = 'Total cost 0', values = df['total_costs_0']),
                                       # dict(label = 'Penalty', values = df['penalty_in'])
                                       ]),
                     )
                )
fig.show()

#%% Scatter plot of all values considering flow rate, bill, area
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

#%% Residual load duration curve
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
fig = make_subplots(rows=1)

color_scale = plotly.colors.sequential.Viridis
color_scale = plotly.colors.qualitative.Bold
for i in range(len(rldc.columns)):
    data = rldc[str(i)]
    name = str(i)
    color_index = int(i / len(rldc.columns) * len(color_scale))
    fig.add_trace(go.Scatter(x=rldc.index, 
                              y=data, 
                              name=str(i),
                              hoverinfo="x+y+name",
                              # line=dict(color=color_scale[color_index]),
                              #line=dict(color=color_scale[i % len(color_scale)])
                              line=dict(color=color_scale[int(i * (len(color_scale) - 1) / (len(rldc.columns) - 1))])
                               )
                   )
fig.show()
# offline.plot(fig, filename='res\\Plots\\dynamic_plot.html', auto_open=False)

#%% OPP plot
import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'browser'
import plotly.express as px

df = df.sort_values(by='design_case')
df['design_case'] = df['design_case'].replace([0,1,2,3,4,5],
                                              ['cp_PV','ST','ASHP','PVT_0','PVT_6','PVT_9'])
fig = px.scatter(df, 
                 # y='opp_import', x=df.index, 
                  y='opp_export', x="opp_import", 
                 symbol="r_level", 
                 color="design_case",
                 # color="r_level", 
                 # symbol="design_case",
                 symbol_sequence= [0,1,2,3,4,5],
                 # facet_col="time",
                 labels={"r_level": "r_level", "design_case": "Design Case"},
                  color_discrete_map={"cp_PV": "grey",
                                      "ST": "red",
                                      'ASHP':'Purple',
                                      "PVT_0": "limegreen",
                                      "PVT_6": "teal",
                                      "PVT_9": "darkblue"},
                 # color_discrete_sequence=px.colors.qualitative.Bold,
                 # color_continuous_scale="oxy",
                 # title="OPP import [kW] for all cases")
                  title="OPP export [kW] vs OPP import [kW] for all cases")
fig.update_traces(marker=dict(size=10))
# fig.update_xaxes(range=[-10, 510])  # Adjust the range as needed for the x-axis
fig.update_yaxes(range=[1.9, 3]) 
fig.update_layout(legend=dict(x=0, y=0))
fig.update_layout(legend=dict(yanchor="bottom", y=0.01,
                              xanchor="right", x=0.99))
fig.show()


#%% plotly plots
# import plotly, plotly.graph_objects as go, plotly.offline as offline, plotly.io as pio
# from plotly.subplots import make_subplots
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
# temp_flow.plot(y=['Thp_load_out','Tsh_in','Tdhw_in', 'Thp_load_in'])
# offline.plot(fig,filename='temp.html')
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