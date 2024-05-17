# -*- coding: utf-8 -*-
"""
Plotting the performance map of heat pump

@author: 20181270
"""
import pandas as pd

folder = 'C:\\TRNSYS18\\Tess Models\\SampleCatalogData\\Water-to-WaterHeatPumps\\Normalized\\'
cdata = folder + 'WWHP_Cooling-Normalized.dat'
hdata = folder + 'WWHP_Heating-Normalized.dat'
hdata = folder + 'WWHP_heating_extrapolated.dat'

# hdata = 'C:/Users/20181270/Downloads/WWHP_Heating-Normalized_2kW.dat'
with open(hdata, 'r') as file:
    lines = file.readlines()
#%%
data = []
for line in lines:
    if not line.startswith('!'):
        values = line.strip().split('\t')
        data.append([float(value) if (value.replace('.', '', 1).lstrip('-')).isdigit() else value for value in values if value != ''])
        '''
        if value.replace('.', '', 1).lstrip('-')).isdigit(): # if the list element is number after removing - sign and decimal, then
            float(value)                                     # make a float of the value
        else:                                                # or leave it as the value itself   
            value                                            # this makes the text part as a separate element '''
# temp = data[3]
# element_last = temp[-1].split(' ', 1)
# float_part = float(element_last[0])
# temp[-1] = float_part
# temp.append(element_last[1])
# data[3] = temp
        # temp = []
        # for value in values:
        #     if value.replace('.', '', 1).isdigit():
        #         temp.append(float(value))
        #     elif value != '':
        #         temp.append(value)
        # data.append(temp)

#%%
inputs = {}
for i in range(4):
    key = data[i][-1]
    values = data[i][:-1]
    inputs[key] = values
    
#%%
order_of_keys = ['!Entering Source Temperatures',
                 '!Entering Load Temperatures',
                 '!Normalized Source Flow Rate',
                 '!Normalized Load Flow Rate']
input_rows = []
for load_flow in inputs['!Normalized Load Flow Rates']:
    for source_flow in inputs['!Normalized Source Flow Rates']:
        for t_load in inputs['!Entering Load Temperatures (C)']:
            for t_source in inputs['!Entering Source Temperatures (C)']:
                input_rows.append([t_source, t_load, source_flow, load_flow])
    
#%%
power_data = data[4:] 
power_data = [[j for j in i if not isinstance(j, str)] for i in power_data]
df_data = [row1 + row2 for row1, row2 in zip(input_rows, power_data)]

columns = ['Teva', 'Tload_in','Feva','Fcond','Capacity','Power']

df2 = pd.DataFrame(data=df_data, columns=columns)   
#%% Input file has capacity, power and flow as normalized values
#   Scaling them to my inputs used in TRNSYS
df = df2.copy()
cap_scale = 30000/3600 #in kW
pow_scale = 6000/3600 # in kW
load_flow_scale = 0.4 # in kg/s 
source_flow_scale = 0.4 # in kg/s
df['Capacity'] = df['Capacity'] * cap_scale
df['Power'] = df['Power'] * pow_scale

df['COP'] = df['Capacity']/df['Power']
df['Fcond'] = df['Fcond'] * load_flow_scale
df['Feva'] = df['Feva'] * source_flow_scale
#%%
df['Tcond'] = df['Capacity']*1000/(df['Fcond']* 4186) +  df['Tload_in']
#%% 3D scatter plot
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(df['Tload_in'], df['Teva'], df['COP'], c=df['COP'],cmap='viridis')
ax.set_xlabel('Condensor temperature')
ax.set_ylabel('Evaporator temperature')
ax.set_zlabel('COP')

#%% Surface plot
from scipy.interpolate import griddata
cond='Tload_in'
xi = np.linspace(df['Teva'].min(), df['Teva'].max(), 100)
yi = np.linspace(df[cond].min(), df[cond].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolating data
zi = griddata((df['Teva'],df[cond]), df['COP'], (xi, yi), method='linear')
levels = np.linspace(np. nanmin(zi), np. nanmax(zi), 100)

print('xi min: ' +str(xi.min()) + ', xi max: ' +str(xi.max())+ 
      ', yi min: ' +str(yi.min()) + ', yi max: '+str(yi.max()) + ', zi max: '+str(np. nanmax(zi)))
fig = plt.figure()
ax = plt.axes(projection='3d')

# Surface plot
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', clim=(df['COP'].min(), df['COP'].max()),  edgecolor='none')
ax.set_xlabel('Evaporator Temperature')
ax.set_ylabel('Condensor Temperature')

ax.set_zlabel('COP')
fig.colorbar(surf)


#%% 2D surface plot. Remove levels i fyou dont mind a courser image
# zi = np.nan_to_num(zi, nan=np.nanmin(zi))
levels = np.linspace(np. nanmin(zi), np. nanmax(zi), 100)

plt.figure(figsize=(8, 6))
cp = plt.contourf(xi, yi, zi, levels=levels, cmap='viridis',vmin=df['COP'].min(), vmax=df['COP'].max())  # Use contourf for filled contours
# cp = plt.contourf(xi, yi, zi, levels=levels, cmap='viridis',vmin=1.7, vmax=9.2) 
cbar = plt.colorbar(cp)  # Show a color bar

# cbar.set_ticks([2.1, 9.2])
plt.xlabel('Evaporator inlet temperature [degC]')
plt.ylabel('Condensor inlet temperature [degC]')
plt.title('Performance map of HP')

#%%
def plot_performance_map():
    
    xi = np.linspace(df['Teva'].min(), df['Teva'].max(), 100)
    yi = np.linspace(df['Tload_in'].min(), df['Tload_in'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolating data
    zi = griddata((df['Teva'],df['Tload_in']), df['COP'], (xi, yi), method='linear')
    levels = np.linspace(np. nanmin(zi), np. nanmax(zi), 100)
    fig, ax = plt.subplots(figsize=(10, 8))
    cp = ax.contourf(xi, yi, zi, levels=levels, cmap='viridis', vmin=2.1, vmax=9.2)
    cbar = plt.colorbar(cp, ax=ax)

    ax.set_xlabel('Evaporator inlet temperature [degC]')
    ax.set_ylabel('Condensor inlet temperature [degC]')
    ax.set_title('Performance map of HP')

    return fig, ax

fig,ax = plot_performance_map()
# 
#%%
discrete_df = {} 
for cond in inputs['!Normalized Load Flow Rates']:
    for eva in  inputs['!Normalized Source Flow Rates']:
        discrete_df[str(cond)+'+'+str(eva)] = df2[(round(df2['Feva'],4)==round(eva,4)) & (round(df2['Fcond'],4)==round(cond,4))]

#%%
from sklearn.linear_model import LinearRegression
def extrapolate_map(df):
    X = df[['Teva','Tload_in']].to_numpy()  # Independent variables: Teva and Tload_in
    y_capacity = df['Capacity']  # Dependent variable: Capacity
    y_power = df['Power']  # Dependent variable: Power
    
    # Model fitting
    model_capacity = LinearRegression().fit(X, y_capacity)
    model_power = LinearRegression().fit(X, y_power)
    # Prediction grid
    Teva_new = np.linspace(-5, 26.6667, 10)  # Extended range for Tevaporator
    Tload_in_new = np.linspace(15.5556, 55, 10)  # Extended range for Tload_in
    
    Teva_new = [-5, -1.1111, 4.4444, 10.0, 15.5556, 21.1111, 26.6667]
    Tload_in_new = [15.5556, 26.6667, 37.7778, 48.8889, 55]
    
    # Generate all combinations of Tevaporator and Tload_in within the new ranges
    Teva_grid, Tload_in_grid = np.meshgrid(Teva_new, Tload_in_new)
    X_new = np.column_stack([Teva_grid.ravel(), Tload_in_grid.ravel()])
    
    # Predicting new values
    Capacity_predicted = model_capacity.predict(X_new)
    Power_predicted = model_power.predict(X_new)

    # creating matching length lists for Fcond and Feva
    feva = [df['Feva'].iloc[0]]*len(Power_predicted)
    fcond = [df['Fcond'].iloc[0]]*len(Power_predicted)
    
    # Creating a dataframe for the predicted values
    predicted_df = pd.DataFrame({
        "Teva": X_new[:, 0],
        "Tload_in": X_new[:, 1],
        "Capacity": Capacity_predicted,
        "Power": Power_predicted, 
        "Feva": feva,
        "Fcond": fcond})
    return predicted_df

new_df = {}
for key in discrete_df:
    new_df[key] = extrapolate_map(discrete_df[key])
    new_df[key]['COP'] = new_df[key]['Capacity']/new_df[key]['Power']

extra_df = pd.concat(new_df.values(), ignore_index=True)  # combine all dataframes into one

#%%
df = df.copy()
df  =df.drop(columns=['Tcond'])
one = df[(df['Feva'].round(4)==0.2172) & (df['Fcond'].round(4)==0.2172)]
two = df[(df['Feva'].round(4)==0.4) & (df['Fcond'].round(4)==0.2172)] 
three = df[(df['Feva'].round(4)==0.5086) & (df['Fcond'].round(4)==0.2172)] 
four = df[(df['Feva'].round(4)==0.2172) & (df['Fcond'].round(4)==0.4)]
five = df[(df['Feva'].round(4)==0.4) & (df['Fcond'].round(4)==0.4)] 
six = df[(df['Feva'].round(4)==0.5086) & (df['Fcond'].round(4)==0.4)]
seven = df[(df['Feva'].round(4)==0.2172) & (df['Fcond'].round(4)==0.5086)]
eight = df[(df['Feva'].round(4)==0.4) & (df['Fcond'].round(4)==0.5086)] 
nine = df[(df['Feva'].round(4)==0.5086) & (df['Fcond'].round(4)==0.5086)]

#%%
from sklearn.linear_model import LinearRegression
X = one[['Teva','Tload_in']].to_numpy()  # Independent variables: Tevaporator and Tload_in
y_capacity = one['Capacity']  # Dependent variable: Capacity
y_power = one['Power']  # Dependent variable: Power

# Model fitting
model_capacity = LinearRegression().fit(X, y_capacity)
model_power = LinearRegression().fit(X, y_power)
# Prediction grid
Teva_new = np.linspace(-5, 26.6667, 10)  # Extended range for Tevaporator
Tload_in_new = np.linspace(15.5556, 55, 10)  # Extended range for Tload_in

Teva_new = [-5, -1.1111, 4.4444, 10.0, 15.5556, 21.1111, 26.6667]
Tload_in_new = [15.5556, 26.6667, 37.7778, 48.8889, 55]

# Generate all combinations of Tevaporator and Tload_in within the new ranges
Teva_grid, Tload_in_grid = np.meshgrid(Teva_new, Tload_in_new)
X_new = np.column_stack([Teva_grid.ravel(), Tload_in_grid.ravel()])

# Predicting new values
Capacity_predicted = model_capacity.predict(X_new)
Power_predicted = model_power.predict(X_new)

# Creating a dataframe for the predicted values
predicted_df = pd.DataFrame({
    "Teva": X_new[:, 0],
    "Tload_in": X_new[:, 1],
    "Capacity": Capacity_predicted,
    "Power": Power_predicted
})

plt.xlim(-1.111,26,667)
plt.ylim([15,49])