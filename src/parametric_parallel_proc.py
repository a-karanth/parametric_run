# -*- coding: utf-8 -*-
"""
Running the sobol samples using parallel processing

"""
import subprocess           # to run the TRNSYS simulation
import shutil               # to duplicate the output txt file
import time                 # to measure the computation time
import os 
import multiprocessing as mp
from ModifyType56 import ModifyType56
from PostprocessFunctions import PostprocessFunctions as pf
from Plots import Plots
from PlotGroups import PlotGroups
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams["figure.autolayout"] = True

DT = '6min'

global directory, folder, res_folder
directory = (os.path.dirname(os.path.realpath(__file__)))
folder = '\\src'
res_folder = 'res\\'
starting_label = 86
mod56 = ModifyType56()
mod56.change_r(directory+'\\House.b18', 'r0')
# mod56.change_r('House_internal_heating.b18', 'r0')


#%% reading CSVs with samples
df1 = pd.read_csv(res_folder+'morris_st_sample.csv')
df2 = pd.read_csv(res_folder+'morris_pvt_sample.csv')
dfmorris = pd.concat([df1,df2])
# dfmorris = pd.read_csv(res_folder+'samples_for_testing.csv')
dfmorris.index = np.arange(len(dfmorris))

df = dfmorris.drop_duplicates()
df.index = np.arange(len(df))

#%% preparing variables for parametric run
batt0 = dict(cell_cap=1, ncell=1, chargeI=1, dischargeI=-1, max_batt_in=0.01, max_batt_out=-0.01, dcv=0.01, ccv=125)
batt6 = dict(cell_cap=48, ncell=50, chargeI=15, dischargeI=-15, max_batt_in=7000, max_batt_out=-4250, dcv=90, ccv=125) 
batt9 = dict(cell_cap=28, ncell=140, chargeI=10, dischargeI=-15, max_batt_in=12000, max_batt_out=-9000, dcv=250, ccv=350)

df['file'], df['py_file'] = [None]*len(df), [None]*len(df)
df['coll_eff'], df['pack'] = [None]*len(df), [None]*len(df)
df['batt'], df['py_label'] = [None]*len(df), [None]*len(df)

for i in range(len(df)):

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
def run_parametric(values):
    # shutil.copy(directory+'\House_internal_heating.b18', directory+'\House_internal_heating_copy'+label+'.b18')
    print(values['py_label'])
    
    df = pd.read_csv('list_of_inputs.csv', delimiter=',', header=0)
    new_row = value[['volume', 'flow_rate', 'coll_area','design_case']]
    df.loc[int(value['py_label'])] = new_row
    
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
    location = directory + '\\' + values['file']
    subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe", location, "/h"])
    elapsed_time = time.time() - start_time # Measuring time (end point)
    print(elapsed_time/60)
    
    # 3) Generating the output .txt file name for each of the simulation results (i.e. first one as 001.txt)
    label_no+=1
    t=str(label_no)
    filename_out=t.rjust(3, '0')+'.txt'
    shutil.copy('trnOut_PumpData.txt', filename_out)

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
