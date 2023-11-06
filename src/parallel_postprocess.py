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

#%% Retrieve labels
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
    new_labels = list(set(check_labels)-set(existing_labels)) # newly simulated labels
    labels = [i for i in labels if any(j in i for j in new_labels)] #checks labels that exist, and copies the exact name (including _cp) for all new labels
else:
    existing_res = pd.DataFrame()
    
#%% Set t_start and t_end
DT = '6min'
t_start = datetime(2001,1,1, 0,0,0)
t_end = datetime(2002,1,1, 0,0,0)

#%% Define parallel processing function
os.chdir(directory)
from PostprocessFunctions import PostprocessFunctions as pf
def parallel_pp(label):
    
    if 'cp' in label:
        controls, energy, temp_flow = pf.cal_base_case(directory+trn_folder + label)
        
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
    pl,pe = pf.peak_load(energy)
    #penalty_in, energy = pf.cal_penalty(energy)
    t1 = datetime(2001,1,4, 0,0,0)
    t2 = datetime(2001,1,8, 0,0,0)
    el_bill_jan, gas_bill_jan, el_em_jan, gas_em_jan, spf = pf.cal_week(controls, energy, temp_flow, t1, t2)
    rldc,ldc = pf.cal_ldc(energy)
    opp_im, opp_ex, import_in, export_in = pf.cal_opp(rldc)
    energy_out = {'el_bill':el_bill,
                  'gas_bill':gas_bill,
                  'el_em': el_em,
                  'gas_em': gas_em,
                  'Q2grid':energy_annual['Q2grid'][0],
                  'Qfrom_grid':energy_annual['Qfrom_grid'][0],
                  'Qpv': energy_annual['Qpv'][0],
                  'Qload':energy_annual['Qload'][0],
                  'Q4sh':energy_annual['Qhp4sh'][0],
                  'Q4dhw':energy_annual['Qhp4tank'][0],
                  'Qaux':energy_annual['Qaux_dhw'][0],
                  'peak_load':pl, 
                  'peak_export':pe,
                  'el_bill_jan':el_bill_jan,
                  'gas_bill_jan':gas_bill_jan,
                  'el_em_jan':el_em_jan, 
                  'gas_em_jan':gas_em_jan,
                  'spf_jan':spf,
                  #'penalty_in': penalty_in,
                  'opp_import':opp_im,
                  'opp_export':opp_ex,
                  'import_in':import_in,
                  'export_in':export_in}
    
    print(label)
    return energy_out, rldc, ldc, label
    # return el_bill, gas_bill, el_em, gas_em, energy_out,el_bill_jan,energy_out2, label

#%% get labels from a redo files
# df = pd.read_csv('res\\redo.csv', index_col=0)
# labels = list(df.index)
# labels = [str(i)+'_cp' for i in labels]

#%% multiprocessing that works
# t1 = time.time()
# print(t1)
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

#%% Joblib parallel processing
from joblib import Parallel, delayed

t1 = time.time()
print(t1)
results = Parallel(n_jobs=8)(delayed(parallel_pp)(label) for label in labels)
t2 = time.time()
print((t2-t1)/60)
    #%% exporting the results in a csv
# #    Flattening the results which are a tuple of dictionary of single values 
# #    and dictionaries 

# output = []
# for i in range(len(results)):
#     flat_data = {}
#     row_data = results[i][0]
#     for key, value in row_data.items():
#         if isinstance(value, dict):
#             for sub_key, sub_value in value.items():
#                 flat_data[f"{key}_{sub_key}"] = sub_value  # converts eg, el_bil dictionary to 4 columns of el_bil_0.1, el_bill_0.5 etc
#         else:
#             flat_data[key] = value  # creates a flat dictionary
#     flat_data['label'] = results[i][3] #label is added as another key in the dictionary
#     output.append(flat_data)
    
# output = pd.DataFrame(output)
# output['label'] = output['label'].str.extract('(\d+)').astype(int)
# output=output.sort_values(by='label', ignore_index=True)
# output = output.set_index('label')
# output = pd.concat([existing_res,output])
# # existing_res.loc[output.index] = output  # to replace only values that were calculated from redo.csv
# output.to_csv(res_folder+'sim_results'+'.csv', index='label', index_label='label')

#%% manual apprach for saving results as csv
# very manual apprach
# energy = results[0][4]
# el_bill = results[0][0]
# el_bill = ['el_bill_'+i for i in el_bill]   #adding the label el_bill before each bill value
# list_columns = [el_bill,'gas_bill', 'el_em', 'gas_em',list(energy.keys()),'label']
# list_columns = list(results[0][0].keys())
# columns = []
#    converting the conbunation of dict an dlist, into a list
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

# output['label'] = output['label'].str.extract('(\d+)').astype(int)
# output=output.sort_values(by='label', ignore_index=True)
# output = output.set_index('label')
# output = pd.concat([existing_res,output])
# output.to_csv(res_folder+'sim_results'+'.csv', index='label', index_label='label')

#%% Results using sequential processing
# os.chdir(directory + res_folder)
# results = pd.DataFrame(columns=['el_bill','gas_bill', 'el_em', 'gas_em','energy_annual','label'])
# for i in labels:
#     energy_out, rldc, ldc, label = parallel_pp(str(i))
    #results.loc[i] = el_bill, gas_bill, el_em, gas_em, energy_annual, label

# results['total_costs'] = results['el_bill']+results['gas_bill']
# results['total_emission'] = (results['el_em']+results['gas_em'])/1000
# results.to_csv('sim_results'+'.csv', index=True)

#%% tests
# test = pd.DataFrame({'label':['1','12_cp','14','102','42_cp','65_cp']})
# test['label'] = test['label'].str.extract('(\d+)').astype(int)

#labels = ['1','2','3','4','5','6','7','8']