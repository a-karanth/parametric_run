# -*- coding: utf-8 -*-
"""
Running the sobol samples using parallel processing

"""
import subprocess           # to run the TRNSYS simulation
import shutil               # to duplicate the output txt file
import time                 # to measure the computation time
import os 
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

sys.path.append(os.getcwd())
from ModifyType56 import ModifyType56
from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots
from PlotGroups import PlotGroups

DT = '6min'

global directory, folder, res_folder
directory = (os.path.dirname(os.path.realpath(__file__)))+'\\'
folder = 'src\\'
res_folder = 'res\\'
trn_folder = 'res\\trn\\'
mod56 = ModifyType56()

#%% existing simulations
existing = pd.read_csv('res\\trn\\list_of_inputs.csv',header=0,index_col=0)
keys = existing.columns.values      #to be used to identify rows while adding to list_of_inputs csv

#%% Finding the last available label and calculating the new starting label
op_labels = os.listdir(directory+trn_folder)
avl_labels = []
for i in op_labels:
    if '_temp_flow' in i:
        prefix = i[:-14]
    elif '_control_signal' in i:
        prefix = i[:-19]
    elif '_energy' in i:
        prefix = i[:-11]
    else:
        continue
    avl_labels.append(prefix)
    
avl_labels = np.unique(np.array(avl_labels))
avl_labels = np.array([int(''.join(filter(str.isdigit, s))) for s in avl_labels])
starting_label = avl_labels.max()+1

#%% reading CSVs with samples
new_sim = True
if new_sim:
    df1 = pd.read_csv(res_folder+'morris_sample_st2.csv')
    df2 = pd.read_csv(res_folder+'morris_sample_pvt2.csv')
    df3 = pd.read_csv(res_folder+'morris_sample_st.csv')
    df4 = pd.read_csv(res_folder+'morris_sample_pvt.csv')
    dfnew = pd.concat([df1,df2,df3,df4,
                       pd.read_csv(res_folder+'morris_sample_cp2.csv'),
                       pd.read_csv(res_folder+'morris_sample_batt1.csv'),
                       pd.read_csv(res_folder+'morris_sample_cp.csv'),
                       pd.read_csv(res_folder+'morris_sample_cp3.csv'),
                       pd.read_csv(res_folder+'morris_sample_pvt4.csv'),
                       pd.read_csv(res_folder+'morris_sample_st4.csv')], ignore_index=True)
    # dfnew = pd.read_csv(res_folder+'morris_sample_cp.csv')
    # dfmorris = pd.read_csv(res_folder+'samples_for_testing.csv')
    # dfnew.index = np.arange(len(dfnew))
    
    dfnew = dfnew.drop_duplicates(ignore_index=True)
    
    df=pd.merge(dfnew, existing, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df.drop(columns=['_merge'], inplace=True)
    df.index = np.arange(len(df))
    save = True
    if save:
        df.to_csv(res_folder+'current_list.csv', index=True, index_label='label')

else:
    df = pd.read_csv('res\\missed_sims2.csv', index_col=0)
    starting_label = 0

#%% preparing variables for parametric run
batt0 = dict(cell_cap=1, ncell=1, chargeI=1, dischargeI=-1, max_batt_in=0.01, max_batt_out=-0.01, dcv=0.01, ccv=125)
batt6 = dict(cell_cap=48, ncell=50, chargeI=15, dischargeI=-15, max_batt_in=7000, max_batt_out=-4250, dcv=90, ccv=125) 
batt9 = dict(cell_cap=28, ncell=140, chargeI=10, dischargeI=-15, max_batt_in=12000, max_batt_out=-9000, dcv=250, ccv=350)

df['file'], df['py_file'] = [None]*len(df), [None]*len(df)
df['coll_eff'], df['pack'] = [None]*len(df), [None]*len(df)
df['batt'], df['py_label'] = [None]*len(df), [None]*len(df)

for i in df.index:

    match df['design_case'][i]:
        case 'ST':  
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.8
            df['pack'][i] = 0
            df['batt'][i] = batt0
        
        case 'PVT':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt0
    
        case 'PVT_Batt_6':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt9
        
        case 'PVT_Batt_9':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt9
        
        case 'cp':
            df['file'][i] = 'wwhp_cp.dck'
            df['py_file'][i] = 'zpy_wwhp_cp.dck'
            df['coll_eff'][i] = 0.05
            df['pack'][i] = 0
            df['batt'][i] = batt0
        
        case 'cp_PV':
            df['file'][i] = 'wwhp_cp.dck'
            df['py_file'][i] = 'zpy_wwhp_cp.dck'
            df['coll_eff'][i] = 0.05
            df['pack'][i] = 0.7
            df['batt'][i] = batt0
     
    df['py_label'][i] = str(starting_label+i)  
    
#%% define parametric run function
os.chdir(directory)
def run_parametric(values):
    # shutil.copy(directory+'\House_internal_heating.b18', directory+'\House_internal_heating_copy'+label+'.b18')
    mod56.change_r(directory+'House.b18', values['r_level'])
    print(values['py_label'])
    
    df = pd.read_csv('res\\trn\\list_of_inputs.csv', header=0, index_col=0)
    new_row = values[keys]
    df.loc[int(values['py_label'])] = new_row
    df.to_csv('res\\trn\\list_of_inputs.csv', index=True, index_label='label')
    
    label_no=0
    with open(values['py_file'], 'r') as file_in:
        filedata = file_in.read()
    
    # filedata = filedata.replace('py_house', label)
    # filedata = filedata.replace('py_dir', directory)
    filedata = filedata.replace('tstart', str(0))
    filedata = filedata.replace('tstop', str(8760))
    filedata = filedata.replace('py_step', '6/60')
    filedata = filedata.replace('py_label', values['py_label'])
    filedata = filedata.replace('py_area_coll', str(values['coll_area']))
    filedata = filedata.replace('py_coll_eff', str(values['coll_eff']))
    filedata = filedata.replace('py_pack', str(values['pack']))
    
    filedata = filedata.replace('py_cell_cap', str(values.batt['cell_cap']))
    filedata = filedata.replace('py_ncell', str(values.batt['ncell']))
    filedata = filedata.replace('py_charge_current', str(values.batt['chargeI']))
    filedata = filedata.replace('py_discharge_current', str(values.batt['dischargeI']))
    filedata = filedata.replace('py_max_batt_in', str(values.batt['max_batt_in']))
    filedata = filedata.replace('py_max_batt_out', str(values.batt['max_batt_out']))
    filedata = filedata.replace('py_dcv', str(values.batt['dcv']))
    filedata = filedata.replace('py_ccv', str(values.batt['ccv']))
    
    filedata = filedata.replace('py_vol', str(values['volume']))
    filedata = filedata.replace('py_flow', str(values['flow_rate']))
    filedata = filedata.replace('py_inf', str(1))
    filedata = filedata.replace('py_db_low', str(2))
    filedata = filedata.replace('py_db_high', str(10))

    #  - (over)writing the modified template .dck file to the original .dck file (to be run by TRNSYS) 
    with open(values['file'], 'w') as dckfile_out:
        dckfile_out.write(filedata)
    # 2) Running TRNSYS simulation
    start_time=time.time()                  # Measuring time (start point)
    location = directory + values['file']
    subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe", location, "/h"])
    elapsed_time = time.time() - start_time # Measuring time (end point)
    print(elapsed_time/60)
    
    # 3) Generating the output .txt file name for each of the simulation results (i.e. first one as 001.txt)
    label_no+=1
    t=str(label_no)
    filename_out=t.rjust(3, '0')+'.txt'
    shutil.copy('trnOut_PumpData.txt', filename_out)

    return 1


def trial_parallel(i):
    df = pd.DataFrame(columns=['file number'])
    df.loc[0] = i
    df.to_csv('trial_'+str(i)+'.csv')
    return 1
#%% Run multiprocessing
t1 = time.time()
print(t1)

# multiprocessing that works
if __name__ == "__main__":
    pool = mp.Pool(8)
    results = []

    for i in range(len(df)):
        time.sleep(15)  # Delay of 15 seconds
        value = df.iloc[i]
        result = pool.apply_async(run_parametric, (value,))
        # result = pool.apply_async(trial_parallel, (i,))
        results.append(result)

    pool.close()
    pool.join()
    
    # Wait for the multiprocessing tasks to complete
    for result in results:
        result.get()
       
t2 = time.time()
print((t2-t1)/60)

#%% Run if multiprocessing is not needed
# no multiprocessing
# for i in range(len(df)):
#     t1 = time.time()
#     value = df.iloc[i]
#     print(value['py_label'])
#     run_parametric(value)
#     t2 = time.time()
#     print((t2-t1)/60)
# pp = pf(DT)
# t_start = datetime(2001,1,1, 0,0,0)
# t_end = datetime(2002,1,1, 0,0,0)
