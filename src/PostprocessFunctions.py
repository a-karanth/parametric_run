# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:45:46 2023

@author: 20181270
"""

import os
import numpy as np
import pandas as pd
import math
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 0.8
matplotlib.rcParams["figure.autolayout"] = True

class PostprocessFunctions:
    
    global DT, dt
    # def __init__():
    #     global DT,dt
      
    @staticmethod    
    def modify_df(df2, t_start, t_end):
        global DT,dt
        # to identify headers from txt file and use them as DF headers
        # to add timestamp column for one year, and to remove the last timestamp 31-12-2001 24:00:00
        # if you have used timestep as 0.125 in trnsys, do not use freq. instead use
        # periods = len(df)-1. Note that with this method, you get decimals in the seconds field. 
        df=df2.copy()
        timestep = int((df.index[2] - df.index[1])*60)
        dt = timestep/60
        DT = str(timestep)+'min'
        
        headers = [i.strip() for i in df.columns]
        df.columns = headers
        df= df.drop(columns = df.columns[-1])
        ts = pd.date_range(start=str(t_start), end=str(t_end), freq=DT )
        df.insert(loc=0, column='Time', value=ts)
        df = df.set_index('Time')
        return df
    
    @staticmethod
    def cal_energy(energy, controls):
        energy['SOC'] = energy['SOC']*3600
        energy['COP'] = energy['COP']*3600
        # if 'batt' not in prefix:
        #     energy['SOC']=0
        energy['Qhp4dhw'] = controls['ctr_dhw']*energy['Qhp']
        energy['Qhp4irr'] = controls['ctr_irr']*energy['Qhp']
        energy['Qhp4tank'] = energy['Qhp4dhw'] + energy['Qhp4irr']
        #calculate energy for space heating. if heat pump is on AND (DHW AND Irr = 0)
        q_sh = [energy.iloc[i]['Qhp'] if (controls.iloc[i]['ctr_dhw']==0 and 
                                          controls.iloc[i]['ctr_irr']==0) else 0 for i in range(len(energy))]
        energy['Qhp4sh'] = q_sh
        energy['gas'] = 0
        energy['Qdev'] = energy[['dev_living','dev_living2' , 'dev_kit', 'dev_bed1', 'dev_bed2',
                                 'dev_bed3', 'dev_attic', 'dev_entree', 'dev_bath']].sum(axis = 1, skipna = True)
        energy['Qltg'] = energy[['ltg_living', 'ltg_living2', 'ltg_kit', 'ltg_bed1', 'ltg_bed2',
                                 'ltg_bed3', 'ltg_attic', 'ltg_entree', 'ltg_bath']].sum(axis = 1, skipna = True)
        energy['Qheat'] = energy['Qhp']*energy['COP']
        # for older results that did not have correct connections to printer
        #energy['Qdev'] = energy[['dev_living', 'dev_kit', 'dev_bed1', 'dev_bed2',
        #                         'dev_bed3', 'dev_attic']].sum(axis = 1, skipna = True)
        #energy['Qltg'] = energy[['ltg_living', 'ltg_kit', 'ltg_bed1', 'ltg_bed2',
        #                         'ltg_bed3', 'ltg_attic']].sum(axis = 1, skipna = True)       
        return energy
    
    @staticmethod
    def create_dfs(prefix):
        t_start = datetime(2001,1,1, 0,0,0)
        t_end = datetime(2002,1,1, 0,0,0)
        temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter = ",", index_col=0)
        energy = pd.read_csv(prefix+'_energy.txt', delimiter = ",", index_col=0)
        controls = pd.read_csv(prefix+'_control_signal.txt', delimiter = ",", index_col=0)
        
        controls = PostprocessFunctions.modify_df(controls, t_start, t_end)
        temp_flow = PostprocessFunctions.modify_df(temp_flow, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        energy = PostprocessFunctions.cal_energy(energy, controls)
        return controls, energy, temp_flow
    
    
    def create_base_dfs(controls, energy, temp_flow, t_start, t_end):
        controls = PostprocessFunctions.modify_df(controls, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy, t_start, t_end)/3600
        energy = PostprocessFunctions.cal_energy(energy, controls)
        temp_flow = PostprocessFunctions.modify_df(temp_flow, t_start, t_end)
        controls = PostprocessFunctions.cal_control(controls, energy, temp_flow)
        return controls, energy, temp_flow
        
    @staticmethod
    def cal_control(control2, energy, temp_flow):
        control = control2.copy()
        control['unmetDHW_wo_aux'] = control['dhw_load_signal']*(temp_flow['T1_dhw']<44.9)
        control['unmetDHW_w_aux'] = control['dhw_load_signal']*(temp_flow['Tat_tap']<44.9)
        return control
    
    @staticmethod
    def cal_integrals(energy):
        global dt
        print('DT = '+ str(dt))
        #calculate monthly and yearly. and drop the last row which is for the next year
        energy_monthly = energy.resample('M').sum()*dt
        energy_annual = energy.resample('Y').sum()*dt
        energy_monthly.drop(index=energy_monthly.index[-1], axis=0, inplace=True)
        energy_annual.drop(index=energy_annual.index[-1], axis=0, inplace=True)
        return energy_monthly, energy_annual
    
    @staticmethod
    def summarize_results(control, energy, temp_flow):
        avg_cop = energy['COP'][energy['COP']!=0].mean()
        unmet_dhw = (control['dhw_load_signal']*(temp_flow['Tat_tap']<44.5)).sum()*100/len(temp_flow)
        unmet_floor1 = ((temp_flow['Tfloor1']<temp_flow['Tset1']-2.5)*1).sum()*100/len(temp_flow)
        unmet_floor2 = ((temp_flow['Tfloor2']<temp_flow['Tset2']-2.5)*1).sum()*100/len(temp_flow)
        h = np.histogram(energy['COP'][energy['COP']!=0], bins=20)
        max_occ_cop = h[1][np.argmax(h[0])]
        rad1 = round(energy['Qrad1'].sum()*dt,2)
        rad2 = round(energy['Qrad2'].sum()*dt,2)
        print('Average COP = ' + str(round(avg_cop,2)) +'\t'+ 
              'Max occured COP = ' + str(round(max_occ_cop,2)))
        print('HP power = ' + str(round(energy['Qhp'].sum()*dt,2))+' kWh \t' +
              'HP power in heating mode = '+ str(round(energy['Qhp'].sum()*dt,2)) + ' kWh')
        print('Unmet DHW = ' + str(round(unmet_dhw,2)) + '%')
        print('Unmet Floor 1 = ' + str(round(unmet_floor1,2)) + '% \t' +   
              'Unmet Floor 2 = ' + str(round(unmet_floor2,2)) + '%')
        print('Qrad1: ' + str(rad1) + ' kWh\t Qrad2: ' + str(rad2) + ' kWh')
        print('Total= '+ str(round(rad1+rad2,2))+ ' kWh')
    
    @staticmethod
    def plot_specs(ax, xlim1=None, xlim2=None, ylim1=0, ylim2=None, ylabel='', 
                   sharedx=True, legend=True, legend_loc='best', title=None, xlabel=0, ygrid=None):
        ax.set_xlim([xlim1, xlim2])
        ax.set_ylim([ylim1, ylim2])
        ax.set_title(title)
                
        if legend ==True:
            ax.legend(loc=legend_loc)
        if xlabel==0:
            ax.xaxis.set_label_text('foo')
            ax.xaxis.label.set_visible(False)
        else:
            ax.set_xlabel(xlabel)
            
        if ylabel=='t':
            ax.set_ylabel('Temperature [deg C]')
        elif ylabel =='p':
            ax.set_ylabel('Power [kW]')
        elif ylabel =='irr':
            ax.set_ylabel('Irradiance [kW/m2]')
        elif ylabel == 'f':
            ax.set_ylabel('Flow rate [kg/hr]')
        else:
            ax.set_ylabel(ylabel)
        
        ax.grid(which='major',  axis='x', linestyle='--', alpha=0.4)  
        if ygrid==True:
            ax.grid(axis='y', linestyle='--', alpha=0.4, lw=0.6)
        elif ygrid=='both':
            ax.grid(axis='both', linestyle='--', alpha=0.4, lw=0.6)
            
    def cal_only_hp(prefix):
        t_start = datetime(2001,1,1, 0,0,0)
        t_end = datetime(2002,1,1, 0,0,0)

        temp_flow = pd.read_csv(prefix+'temp_flow.txt', delimiter = ",", index_col=0)
        energy = pd.read_csv(prefix+'energy.txt', delimiter = ",", index_col=0)
        controls = pd.read_csv(prefix+'control_signal.txt', delimiter = ",", index_col=0)
        
        controls = PostprocessFunctions.modify_df(controls, t_start, t_end)
        temp_flow = PostprocessFunctions.modify_df(temp_flow, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        energy = PostprocessFunctions.cal_energy(energy, controls, prefix)
        
        energy['Qheat'] = energy['Qheat_living']+energy['Qheat_bed1']+energy['Qheat_bed2']
        energy['Qhp4sh'] = energy['Qheat']/2.8
        energy['Qhp4tank'] = energy['Qaux_dhw']/1.9
        energy['Qaux_dhw'] = 0
        energy['Qload'] = energy['Qdev']+energy['Qltg']+energy['Qhp4sh']+energy['Qhp4tank']
        energy['Qfrom_grid'] = energy['Qload']
        
        energy_monthly, energy_annual = PostprocessFunctions.cal_integrals(energy)
        
        energyH = (energy.resample('H').sum())*dt
        rldc = pd.DataFrame()
        rldc['net_import']=-energyH['Q2grid']+energyH['Qfrom_grid']            
        rldc = rldc.sort_values(by=['net_import'], ascending = False)
        rldc['interval']=1
        rldc['duration'] = rldc['interval'].cumsum()
        
        ldc = pd.DataFrame()
        ldc['load'] = energyH['Qload']
        ldc = ldc.sort_values(by=['load'], ascending=False)
        ldc['interval'] = 1
        ldc['duration'] = ldc['interval'].cumsum()
        
        results = dict(controls=controls, 
                       energy=energy, 
                       temp_flow=temp_flow, 
                       energy_monthly=energy_monthly,
                       rldc=rldc,
                       ldc=ldc)

        return controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc
    
    def cal_base_case(prefix):
        # without HP
        print('cp')
        t_start = datetime(2001,1,1, 0,0,0)
        t_end = datetime(2002,1,1, 0,0,0)

        temp_flow = pd.read_csv(prefix+'_temp_flow.txt', delimiter = ",", index_col=0)
        energy = pd.read_csv(prefix+'_energy.txt', delimiter = ",", index_col=0)
        controls = pd.read_csv(prefix+'_control_signal.txt', delimiter = ",", index_col=0)
        
        controls = PostprocessFunctions.modify_df(controls, t_start, t_end)
        temp_flow = PostprocessFunctions.modify_df(temp_flow, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy, t_start, t_end)/3600     # kJ/hr to kW 
        
        energy = PostprocessFunctions.cal_energy(energy, controls)
        
        # energy['Qheat'] = energy['Qheat_living1']+ energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']+energy['Qaux_dhw']
        # energy['Qhp4sh'] = energy['Qheat_living1']+ energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']
        energy['Qheat'] = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']+energy['Qaux_dhw']
        energy['Qhp4sh'] = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']
        energy['Qhp4tank'] = energy['Qaux_dhw']
        energy['Qhp'] = 1 # necessary to have a non zero value during the calculation of SPF
        energy['Qaux_dhw'] = 0
        energy['gas'] = energy['Qheat']/10
        
        energy_monthly, energy_annual = PostprocessFunctions.cal_integrals(energy)
        
        energyH = (energy.resample('H').sum())*dt #hourly energy
        rldc = pd.DataFrame()
        rldc['net_import']=-energyH['Q2grid']+energyH['Qfrom_grid']            
        rldc = rldc.sort_values(by=['net_import'], ascending = False)
        rldc['interval']=1
        rldc['duration'] = rldc['interval'].cumsum()
        
        ldc = pd.DataFrame()
        ldc['load'] = energyH['Qload']
        ldc = ldc.sort_values(by=['load'], ascending=False)
        ldc['interval'] = 1
        ldc['duration'] = ldc['interval'].cumsum()
        
        results = {'control': controls,
                   'energy': energy,
                   'temp_flow': temp_flow,
                   'energy_monthly': energy_monthly,
                   'rldc': rldc,
                   'ldc': ldc}
        return controls, energy, temp_flow, energy_monthly, energy_annual, rldc, ldc
    
    def cal_costs(energy, nm=[1,0.5,0.1,0], el_cost=0.4, feedin_tariff=0.07, gas_cost=1.45):
        global dt
        # returns energy bill in EUR. 
        # Costs are in EUR/kWh for electricity, EUR/m3 for gas
        el_import = math.ceil(energy['Qfrom_grid'].sum()*dt)

        el_bill={}
        for n in nm:     
            energy['export'] = energy['Q2grid'] * n
            energy['excess'] = energy['Q2grid'] * (1-n)
            el_export = math.ceil(energy['export'].sum()*dt)
            el_excess = math.ceil(energy['excess'].sum()*dt)
            if el_import >= el_export:
                net_metered_cost = (el_import - el_export) * el_cost
                excess_cost = el_excess * feedin_tariff
            else:
                net_metered_cost = el_export * feedin_tariff
                excess_cost = (el_export - el_import) * feedin_tariff
            el_bill[str(n)] = net_metered_cost + excess_cost
    
        gas_bill = (energy['gas']*gas_cost).sum()*dt
        return el_bill, gas_bill
    
    def cal_emissions(energy, el_ef=340, gas_ef=2016):
        # returns emissions in gCO2, 
        # emission factor in gCO2/kWh for electricity, gCO2/m3 for gas
        aef = pd.read_csv('AEF_annual.csv', index_col=0)
        aef.columns=aef.columns.map(int)
        energy['aef']=0
        for index, imp in energy['aef'].items():
            energy['aef'].loc[index] = aef[index.month][index.hour]
        el_em = ((energy['Qfrom_grid']*energy['aef']).sum()*dt)/1000
        gas_em = ((energy['gas']*gas_ef).sum()*dt)/1000
        return el_em, gas_em
    
    def peak_load(energy):
        # peak load, peak import
        pl = energy['Qload'].max()
        pe = energy['Q2grid'].max()
        return pl,pe
    
    def cal_week(controls, energy, temp_flow, t1,t2):
        global dt
        energy = energy[t1:t2]
        el_bill, gas_bill = PostprocessFunctions.cal_costs(energy)
        el_em, gas_em = PostprocessFunctions.cal_emissions(energy)
        spf = PostprocessFunctions.cal_spf(energy)
        return el_bill, gas_bill, el_em, gas_em, spf
    
    def cal_penalty(energy):
        global dt
        pen = pd.read_csv('penalty_inverse.csv', index_col=0)
        pen.columns=pen.columns.map(int)
        energy['pen']=0
        for index, value in energy['pen'].items():
            energy['pen'].loc[index] = pen[index.month][index.hour]
        energy['penalty'] = energy['Q2grid']*energy['pen']
        penalty = energy['penalty'].sum()*dt
        return penalty, energy
    
    def cal_spf(energy):
        spf = energy['Qheat'].sum()/energy['Qhp'].sum()
        return spf