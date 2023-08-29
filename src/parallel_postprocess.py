# %%
# -*- coding: utf-8 -*-
"""
post processing results using parallel processing
collate results into a csv for later use

@author: 20181270
"""

import time                 # to measure the computation time
import os 
import multiprocessing as mp
# from PostprocessFunctions import PostprocessFunctions as pf
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

global directory, result_folder
directory = 'C:\\Users\\20181270\\OneDrive - TU Eindhoven\\PhD\\TRNSYS\\Publication1\\pub_1\\src\\'
res_folder = 'res\\'
trn_folder = 'res\\trn\\'

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
check_labels = np.array([''.join(filter(str.isdigit, s)) for s in labels])

#%% check if sim_results.csv exists. create if it doesnt, add new values to it, if it does
sim_yn =  os.listdir(directory+res_folder)
if 'sim_results.csv' in sim_yn:
    existing_res = pd.read_csv(directory+res_folder + 'sim_results.csv',index_col='label')
    existing_labels = np.array(existing_res.index.astype(str).tolist())
    new_labels = list(set(check_labels)-set(existing_labels))
    labels = [i for i in labels if any(j in i for j in new_labels)]
else:
    existing_res = pd.DataFrame()
#%%
DT = '6min'
t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

#%%
os.chdir(directory)
from PostprocessFunctions import PostprocessFunctions as pf
def parallel_pp(label):
    
    if 'cp' in label:
        controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc = pf.cal_base_case(directory+trn_folder + label)
        
    else:
        temp_flow = pd.read_csv(directory+trn_folder + label+'_temp_flow.txt', delimiter=",",index_col=0)
        energy = pd.read_csv(directory+trn_folder + label+'_energy.txt', delimiter=",", index_col=0)
        controls = pd.read_csv(directory+trn_folder + label+'_control_signal.txt', delimiter=",",index_col=0)
        
        controls = pf.modify_df(controls, t_start, t_end)
        temp_flow = pf.modify_df(temp_flow, t_start, t_end)
        energy = pf.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        energy = pf.cal_energy(energy, controls)
        energy_monthly, energy_annual = pf.cal_integrals(energy)
        
    el_bill, gas_bill = pf.cal_costs(energy)
    el_em, gas_em = pf.cal_emissions(energy)
    energy_out = {'Q2grid':energy_annual['Q2grid'][0],
                  'Qfrom_grid':energy_annual['Qfrom_grid'][0],
                  'Qpv': energy_annual['Qpv'][0],
                  'Qload':energy_annual['Qload'][0],
                  'SOC':energy_annual['SOC'][0]}
    print(label)
    return el_bill, gas_bill, el_em, gas_em, energy_out, label

#%% Results using sequential processing
# os.chdir(directory + res_folder)
# results = pd.DataFrame(columns=['el_bill','gas_bill', 'el_em', 'gas_em','energy_annual','label'])
# count = 0
# for i in labels:
#     el_bill, gas_bill, el_em, gas_em, energy_annual, label = parallel_pp(str(i))
#     results.loc[i] = el_bill, gas_bill, el_em, gas_em, energy_annual, label
#     count = count+1
#     print(count)

# results['total_costs'] = results['el_bill']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000
# results.to_csv('sim_results'+'.csv', index=True)

#%%
# # multiprocessing that works
t1 = time.time()
if __name__ == "__main__":
    pool = mp.Pool(8)
    results = pool.map(parallel_pp, labels)
    
    pool.close()
    pool.join()
    # results = []

    # for i in range(len(labels)):
        # time.sleep(3)  # Delay of 15 seconds
        # result = pool.apply_async(parallel_pp, (str(i),))
        # results.append(result)
    # output = [result.get() for result in results]
    
    # Wait for the multiprocessing tasks to complete
    # for result in results:
    #     result.get()
        
t2 = time.time()
print(t2-t1)

#%% exporting the results in a csv
##    isolating the results that come as dictionaries: annual energy, electricity 
##    bill for different net metering levels
# energy = results[0][4]
# el_bill = results[0][0]
# el_bill = ['el_bill_'+i for i in el_bill]   #adding the label el_bill before each bill value
# list_columns = [el_bill,'gas_bill', 'el_em', 'gas_em',list(energy.keys()),'label']
# columns = []
##    converting the conbunation of dict an dlist, into a list
# for item in list_columns:
#     if isinstance(item, list):
#         columns.extend(item)
#     else:
#         columns.append(item)
        
# output = pd.DataFrame(columns=columns)        
# for i in range(len(results)):
#     row = []
#     row_data = results[i]
#     for r in row_data:
#         if isinstance(r,dict):
#             row.extend(r.values())
#         else:
#             row.append(r)
#     output.loc[i] = row


# # # output['label']=output['label'].astype(int)
# output['label'] = output['label'].str.extract('(\d+)').astype(int)
# output=output.sort_values(by='label', ignore_index=True)
# output = output.set_index('label')
# output = pd.concat([existing_res,output])
# output.to_csv(res_folder+'sim_results'+'.csv', index='label', index_label='label')

#%% tests
# test = pd.DataFrame({'label':['1','12_cp','14','102','42_cp','65_cp']})
# test['label'] = test['label'].str.extract('(\d+)').astype(int)