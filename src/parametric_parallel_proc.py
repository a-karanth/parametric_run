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

global directory, folder
directory = (os.path.dirname(os.path.realpath(__file__)))
folder = '\\with_summer_loop'
mod56 = ModifyType56()
mod56.change_r('House_copy.b18', 'r0')
# mod56.change_r('House_internal_heating.b18', 'r0')

os.chdir(directory + folder)

dfsobol = pd.read_csv('Sobol_samples2_missed_sims.csv')
dfsobol = dfsobol.drop_duplicates()
dfsobol.index = np.arange(len(dfsobol))

batt0 = dict(cell_cap=1, ncell=1, chargeI=1, max_batt_in=0.01, max_batt_out=-0.01, dcv=0.01, ccv=125)
batt6 = dict(cell_cap=48, ncell=50, chargeI=15, max_batt_in=7000, max_batt_out=-4250, dcv=90, ccv=125) 
batt9 = dict(cell_cap=28, ncell=140, chargeI=10, max_batt_in=12000, max_batt_out=-9000, dcv=250, ccv=350)

dfsobol['file'], dfsobol['py_file'] = [None]*len(dfsobol), [None]*len(dfsobol)
dfsobol['coll_eff'], dfsobol['pack'] = [None]*len(dfsobol), [None]*len(dfsobol)
dfsobol['batt'], dfsobol['py_label'] = [None]*len(dfsobol), [None]*len(dfsobol)

for i in range(len(dfsobol)):

    match dfsobol['design_case'][i]:
        case 'ST':  
            dfsobol['file'][i] = 'wwhp.dck'
            dfsobol['py_file'][i] = 'zpy_wwhp.dck'
            dfsobol['coll_eff'][i] = 0.8
            dfsobol['pack'][i] = 0
            dfsobol['batt'][i] = batt0
        
        case 'PVT':
            dfsobol['file'][i] = 'wwhp.dck'
            dfsobol['py_file'][i] = 'zpy_wwhp.dck'
            dfsobol['coll_eff'][i] = 0.7
            dfsobol['pack'][i] = 0.7
            dfsobol['batt'][i] = batt0
    
        case 'PVT_Batt_6':
            dfsobol['file'][i] = 'wwhp.dck'
            dfsobol['py_file'][i] = 'zpy_wwhp.dck'
            dfsobol['coll_eff'][i] = 0.7
            dfsobol['pack'][i] = 0.7
            dfsobol['batt'][i] = batt9
        
        case 'PVT_Batt_9':
            dfsobol['file'][i] = 'wwhp.dck'
            dfsobol['py_file'][i] = 'zpy_wwhp.dck'
            dfsobol['coll_eff'][i] = 0.7
            dfsobol['pack'][i] = 0.7
            dfsobol['batt'][i] = batt9
        
        case 'base':
            dfsobol['file'][i] = 'cp.dck'
            dfsobol['py_file'][i] = 'zpy_cp.dck'
            dfsobol['coll_eff'][i] = 0.05
            dfsobol['pack'][i] = 0
            dfsobol['batt'][i] = batt0
        
        case 'base_PV':
            dfsobol['file'][i] = 'cp.dck'
            dfsobol['py_file'][i] = 'zpy_cp.dck'
            dfsobol['coll_eff'][i] = 0.05
            dfsobol['pack'][i] = 0.7
            dfsobol['batt'][i] = batt0
     
    dfsobol['py_label'][i] = dfsobol['design_case'][i]+'_V'+str(dfsobol['volume'][i]).replace('.','_')+'_A'+str(dfsobol['coll_area'][i])  

def run_parametric(values):
    # shutil.copy(directory+'\House_internal_heating.b18', directory+'\House_internal_heating_copy'+label+'.b18')
    print(values['py_label'])
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
    filedata = filedata.replace('py_max_batt_in', str(values.batt['max_batt_in']))
    filedata = filedata.replace('py_max_batt_out', str(values.batt['max_batt_out']))
    filedata = filedata.replace('py_dcv', str(values.batt['dcv']))
    filedata = filedata.replace('py_ccv', str(values.batt['ccv']))
    
    filedata = filedata.replace('py_vol', str(values['volume']))

    #  - (over)writing the modified template .dck file to the original .dck file (to be run by TRNSYS) 
    with open(values['file'], 'w') as dckfile_out:
        dckfile_out.write(filedata)
    # 2) Running TRNSYS simulation
    start_time=time.time()                  # Measuring time (start point)
    location = directory + folder + '\\' + values['file']
    subprocess.run([r"C:\TRNSYS18\Exe\TrnEXE64.exe", location, "/h"])
    elapsed_time = time.time() - start_time # Measuring time (end point)
    print(elapsed_time/60)
    
    # 3) Generating the output .txt file name for each of the simulation results (i.e. first one as 001.txt)
    label_no+=1
    t=str(label_no)
    filename_out=t.rjust(3, '0')+'.txt'
    shutil.copy('trnOut_PumpData.txt', filename_out)

    return 1



t1 = time.time()
print(t1)


# multiprocessing that works
if __name__ == "__main__":
    pool = mp.Pool(8)
    results = []

    for i in range(len(dfsobol)):
        time.sleep(15)  # Delay of 15 seconds
        value = dfsobol.iloc[i]
        result = pool.apply_async(run_parametric, (value,))
        results.append(result)

    pool.close()
    pool.join()
    
    # Wait for the multiprocessing tasks to complete
    for result in results:
        result.get()
       
# t2 = time.time()
# print((t2-t1)/60)


# no multiprocessing
# for i in range(len(dfsobol)):
#     t1 = time.time()
#     value = dfsobol.iloc[i]
#     print(value['py_label'])
#     run_parametric(value)
#     t2 = time.time()
#     print((t2-t1)/60)
# pp = pf(DT)
# t_start = datetime(2001,1,1, 0,0,0)
# t_end = datetime(2002,1,1, 0,0,0)
