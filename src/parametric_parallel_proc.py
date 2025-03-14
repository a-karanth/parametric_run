# -*- coding: utf-8 -*-
"""
Running the sobol samples using parallel processing

"""
import subprocess           # to run the TRNSYS simulation
import shutil               # to duplicate the output txt file
import time                 # to measure the computation time
import os 
import sys
os.chdir(os.path.abspath(os.path.dirname(__file__)))
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
existing['coll_area'] = existing['coll_area'].astype(int)
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
if len(avl_labels) ==0:
    starting_label = 0
else:
    starting_label = avl_labels.max()+1

#%% reading CSVs with samples
new_sim = False
if new_sim:
    dfnew = pd.concat([pd.read_csv(res_folder+'small_study.csv'),
                       ])
    dfnew = dfnew.drop_duplicates(ignore_index=True)
    
    df=pd.merge(dfnew, existing, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df.drop(columns=['_merge'], inplace=True)
    df.index = np.arange(len(df))
    save = True
    if save:
        df.to_csv(res_folder+'current_list.csv', index=True, index_label='label')

else:
    df = pd.read_csv('res\\redo.csv', index_col=0)
    starting_label = 0

#%% preparing variables for parametric run
batt0 = dict(cell_cap=1, ncell=1, chargeI=1, dischargeI=-1, max_batt_in=0.01, max_batt_out=-0.01, dcv=0.01, ccv=125)
batt6 = dict(cell_cap=48, ncell=50, chargeI=15, dischargeI=-15, max_batt_in=7000, max_batt_out=-4250, dcv=90, ccv=125) 
batt9 = dict(cell_cap=28, ncell=140, chargeI=10, dischargeI=-15, max_batt_in=12000, max_batt_out=-9000, dcv=250, ccv=350)

df['file'], df['py_file'] = [None]*len(df), [None]*len(df)
df['coll_eff'], df['pack'] = [None]*len(df), [None]*len(df)
df['batt'], df['py_label'] = [None]*len(df), [None]*len(df)
df['house'] = [None]*len(df)
df['draw_folder'] = [None]*len(df)
for i in df.index:

    match df['design_case'][i]:
        case 'ST':  
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.8
            df['pack'][i] = 0
            df['batt'][i] = batt0
            df['house'][i] = 'House.b18'
        
        case 'PVT_0':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt0
            df['house'][i] = 'House.b18'
    
        case 'PVT_6':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt6
            df['house'][i] = 'House.b18'
        
        case 'PVT_9':
            df['file'][i] = 'wwhp.dck'
            df['py_file'][i] = 'zpy_wwhp.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt9
            df['house'][i] = 'House.b18'
            
        case 'ASHP':
            df['file'][i] = 'ashp.dck'
            df['py_file'][i] = 'zpy_ashp.dck'
            df['coll_eff'][i] = 0.05
            df['pack'][i] = 0.7
            df['batt'][i] = batt0
            df['house'][i] = 'House.b18'
        
        case 'cp_PV':
            df['file'][i] = 'wwhp_cp.dck'
            df['py_file'][i] = 'zpy_wwhp_cp.dck'
            df['coll_eff'][i] = 0.05
            df['pack'][i] = 0.7
            df['batt'][i] = batt0
            df['house'][i] = 'House_internal_heating.b18'
            
        case 'PVT_buff':
            df['file'][i] = 'wwhp_buffer3.dck'
            df['py_file'][i] = 'zpy_wwhp_buffer3.dck'
            df['coll_eff'][i] = 0.7
            df['pack'][i] = 0.7
            df['batt'][i] = batt6
            df['house'][i] = 'House.b18'
     
    df['py_label'][i] = str(starting_label+i) 
    
    match df['draw'][i]:
        case 'low':
            df['draw_folder'][i] = 'DHW0002_1'
        case 'medium':
            df['draw_folder'][i] = 'DHW0002_2'
        case 'high':
            df['draw_folder'][i] = 'DHW0002_3'
    
#%% define parametric run function
os.chdir(directory)
def run_parametric(values):
    # shutil.copy(directory+'\House_internal_heating.b18', directory+'\House_internal_heating_copy'+label+'.b18')
    house_file = directory+'house_and_backup\\'+values['house']
    mod56.change_r(house_file, 
                   values['r_level'], 
                   values['inf'],
                   values['py_label'])
    print(values['py_label'])
    
    df = pd.read_csv('res\\trn\\list_of_inputs.csv', header=0, index_col=0)
    new_row = values[keys]
    df.loc[int(values['py_label'])] = new_row
    df.to_csv('res\\trn\\list_of_inputs.csv', index=True, index_label='label')
    
    inp_files = pd.read_csv('res\\trn\\input_files.csv', header=0, index_col=0)
    inp_files.loc[int(values['py_label'])] = [values['house'], values['file'] ,values['py_file'],values['draw_folder']]
    inp_files.to_csv('res\\trn\\input_files.csv', index=True, index_label='label')
    
    label_no=0
    with open(values['py_file'], 'r') as file_in:
        filedata = file_in.read()
    
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
    filedata = filedata.replace('py_flow_factor', str(values['flow_factor']))
    filedata = filedata.replace('py_draw_folder', str(values['draw_folder']))
    filedata = filedata.replace('py_inf', str(1))
    filedata = filedata.replace('py_db_low', str(2))
    filedata = filedata.replace('py_db_high', str(10))

    #  - (over)writing the modified template .dck file to the original .dck file (to be run by TRNSYS) 
    final_dck = values['file'].replace('.dck','_'+values['py_label']+'.dck')
    with open(final_dck, 'w') as dckfile_out:
        dckfile_out.write(filedata)
        
    #  - writing the dck file to a new dck file as back up  
    # with open(directory+'house_and_backup\\backup\\' +
    #           values['file'][:-4] + '_'+values['py_label'] + '.dck', 'w') as backup:
    #     backup.write(filedata)
        
    # 2) Running TRNSYS simulation
    start_time=time.time()                  # Measuring time (start point)
    location = directory + final_dck
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
