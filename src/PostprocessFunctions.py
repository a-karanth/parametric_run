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
    
    global DT, dt, t_start, t_end
    # def __init__():
    #     global DT,dt
      
    @staticmethod    
    def modify_df(df2):#, t_start, t_end):
        global DT, dt, t_start, t_end
        # to identify headers from txt file and use them as DF headers
        # to add timestamp column for one year, and to remove the last timestamp 31-12-2001 24:00:00
        # if you have used timestep as 0.125 in trnsys, do not use freq. instead use
        # periods = len(df)-1. Note that with this method, you get decimals in the seconds field. 
        df=df2.copy()
        timestep = int((df.index[2] - df.index[1])*60)
        dt = timestep/60
        DT = str(timestep)+'min'
        
        t_start = datetime(2001, 1, 1) + timedelta(hours=df.index[0])
        t_end = datetime(2001, 1, 1) + timedelta(hours=df.index[-1])
        headers = [i.strip() for i in df.columns]
        df.columns = headers
        df= df.drop(columns = df.columns[-1])
        ts = pd.date_range(start=str(t_start), end=str(t_end), freq=DT )
        df.insert(loc=0, column='Time', value=ts)
        df = df.set_index('Time')
        return df
    
    @staticmethod
    def extract_params(df2):#, t_start, t_end):
        global DT, dt, t_start, t_end
        # to identify parameters and extract headers and data
        df=df2.copy()
        headers = [i.strip() for i in df.columns]
        df.columns = headers
        df= df.drop(columns = df.columns[-1])
        return df
    
    @staticmethod
    def cal_energy(energy, controls):
        energy['SOC'] = energy['SOC']*3600
        energy['COP'] = energy['COP']*3600
        energy['COP'].replace(0,np.NaN, inplace=True)
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
        energy['Qdev'] = energy[['dev_living1','dev_living2' , 'dev_kitchen', 'dev_bed1', 'dev_bed2',
                                 'dev_bed3', 'dev_attic', 'dev_entree', 'dev_bath']].sum(axis = 1, skipna = True)
        energy['Qltg'] = energy[['ltg_living1', 'ltg_living2', 'ltg_kitchen', 'ltg_bed1', 'ltg_bed2',
                                 'ltg_bed3', 'ltg_attic', 'ltg_entree', 'ltg_bath']].sum(axis = 1, skipna = True)
        energy['Qheat'] = energy['Qhp_load'] + energy['Qaux_hp'] + energy['Qaux_dhw']
        energy['Qcoll_cum'] = energy['QuColl'].cumsum()*0.1
        energy['Qpv_cum'] = energy['Qpv'].cumsum()*0.1
        energy['QuColl_irr'] = energy['QuColl'] * controls['ctr_irr']
        energy['QuColl_t'] = np.where(((controls['ctr_coll_t'] == 1) | (controls['t_inlet_below_ambient'] == 1)) & (controls['ctr_irr'] == 0), energy['QuColl'], 0)  # or any value you want when the condition is not met
        energy['QuColl_pos'] = energy['QuColl'].apply(lambda x: x if x >= 0 else np.nan)
        energy['QuColl_irr_pos'] = energy['QuColl_irr'].apply(lambda x: x if x >= 0 else np.nan)
        energy['QuColl_t_pos'] = energy['QuColl_t'].apply(lambda x: x if x >= 0 else np.nan)
        energy['QuColl_irr_neg'] = energy['QuColl_irr'].apply(lambda x: x if x < 0 else np.nan)
        energy['QuColl_t_neg'] = energy['QuColl_t'].apply(lambda x: x if x < 0 else np.nan)
        energy['QuColl_neg'] = energy['QuColl'].apply(lambda x: x if x < 0 else np.nan)
        energy['Qhx_pos'] = energy['Qhx'].apply(lambda x: x if x >= 0 else np.nan)
        energy['Qhx_neg'] = energy['Qhx'].apply(lambda x: x if x < 0 else np.nan)
        energy['Qhp_source_pos'] = energy['Qhp_source'].apply(lambda x:x if x>=0 else np.nan)
        energy['Qhp_source_neg'] = energy['Qhp_source'].apply(lambda x:x if x<0 else np.nan)
        energy['Qsh_buff_in'] = energy['Qsh_buff_source'].apply(lambda x:x if x<=0 else np.nan)
        energy['Qsh_buff_out'] = energy['Qsh_buff_source'].apply(lambda x:x if x>0 else np.nan)
        energy['Qdhw_in'] = energy['Qdhw_source'].apply(lambda x:x if x<=0 else np.nan)
        energy['Qdhw_out'] = energy['Qdhw_source'].apply(lambda x:x if x>0 else np.nan)
        
    
        if 'Qloss_dhw' in energy.columns:
            energy['Qloss_load'] = energy[['Qloss_dhw','Qloss_sh']].sum(axis=1, skipna=True)
        # for older results that did not have correct connections to printer
        #energy['Qdev'] = energy[['dev_living', 'dev_kit', 'dev_bed1', 'dev_bed2',
        #                         'dev_bed3', 'dev_attic']].sum(axis = 1, skipna = True)
        #energy['Qltg'] = energy[['ltg_living', 'ltg_kit', 'ltg_bed1', 'ltg_bed2',
        #                         'ltg_bed3', 'ltg_attic']].sum(axis = 1, skipna = True)       
        return energy
    
  
    def create_base_dfs(controls, energy, temp_flow):#, t_start, t_end):
        controls = PostprocessFunctions.modify_df(controls)#, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy)#, t_start, t_end)/3600
        energy = PostprocessFunctions.cal_energy(energy, controls)
        temp_flow = PostprocessFunctions.modify_df(temp_flow)#, t_start, t_end)
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
        #calculate monthly and yearly. and drop the last row which is for the next year
        energy_monthly = energy.resample('M').sum()*dt
        energy_annual = energy.resample('Y').sum()*dt
        energy_monthly.drop(index=energy_monthly.index[-1], axis=0, inplace=True)
        energy_annual.drop(index=energy_annual.index[-1], axis=0, inplace=True)
        return energy_monthly, energy_annual
    
    @staticmethod
    def cal_daily_hourly(energy):
        global dt
        energy_daily = energy.resample('D').sum()*dt
        energy_hourly = energy.resample('H').sum()*dt
        # energy_daily['Qstored_dhw'] = energy.resample('D').agg({'Qstored_dhw': 'last'}) # if you want the last value of each day
        return energy_daily, energy_hourly
    
    @staticmethod
    def cal_cumulative(energy):
        global dt
        dt = 0.1
        columns = ['Qheat', 'Qhp', 'Qaux_hp','emission','el_bill']
        cumulative_yearly = energy[columns].cumsum()*dt
    
        # Create cumulative values on a monthly basis
        cumulative_monthly = energy[columns].copy()
        cumulative_monthly['Month'] = cumulative_monthly.index.to_period('M')
        cumulative_monthly = cumulative_monthly.groupby('Month').cumsum()*dt
        # cumulative_monthly.drop(columns='Month', inplace=True)
    
        # Create cumulative values on a daily basis
        cumulative_daily = energy[columns].copy()
        cumulative_daily['Day'] = cumulative_daily.index.to_period('D')
        cumulative_daily = cumulative_daily.groupby('Day').cumsum()*dt
        # cumulative_daily.drop(columns='Day', inplace=True)
        return cumulative_yearly, cumulative_monthly, cumulative_daily
    
    def cumulative_emission_costs(energy):
        columns = ['emission','el_bill']
        cumulative_yearly = energy[columns].cumsum()
        # Create cumulative values on a monthly basis
        cumulative_monthly = energy[columns].copy()
        cumulative_monthly['Month'] = cumulative_monthly.index.to_period('M')
        cumulative_monthly = cumulative_monthly.groupby('Month').cumsum()
        # cumulative_monthly.drop(columns='Month', inplace=True)
    
        # Create cumulative values on a daily basis
        cumulative_daily = energy[columns].copy()
        cumulative_daily['Day'] = cumulative_daily.index.to_period('D')
        cumulative_daily = cumulative_daily.groupby('Day').cumsum()
        # cumulative_daily.drop(columns='Day', inplace=True)
        return cumulative_yearly, cumulative_monthly, cumulative_daily
    
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
    def plot_specs(ax, xlim1=None, xlim2=None, ylim1=None, ylim2=None, ylabel='', 
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
        
        if ygrid==True:
            ax.grid(which='major',  axis='both', linestyle='--', alpha=0.4,lw=0.6)
            ax.grid(which='minor', axis='x', linestyle='--', alpha=0.4, lw=0.6)
            

    def cal_base_case(file):
        # without HP
        global DT, dt, t_start, t_end

        temp_flow = pd.read_csv(file+'_temp_flow.txt', delimiter = ",", index_col=0)
        energy = pd.read_csv(file+'_energy.txt', delimiter = ",", index_col=0)
        controls = pd.read_csv(file+'_control_signal.txt', delimiter = ",", index_col=0)
        
        controls = PostprocessFunctions.modify_df(controls)#, t_start, t_end)
        temp_flow = PostprocessFunctions.modify_df(temp_flow)#, t_start, t_end)
        energy = PostprocessFunctions.modify_df(energy)/3600#, t_start, t_end)/3600     # kJ/hr to kW 
        
        energy = PostprocessFunctions.cal_energy(energy, controls)
        
        energy['Qheat'] = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']+energy['Qaux_dhw']
        energy['Qhp4sh'] = energy['Qheat_living1']+energy['Qheat_living2']+energy['Qheat_bed1']+energy['Qheat_bed2']
        energy['Qhp4tank'] = energy['Qaux_dhw']
        energy['Qhp'] = 1 # necessary to have a non zero value during the calculation of SPF
        energy['Qaux_dhw'] = 0
        energy['gas'] = energy['Qheat']/10
        
        return controls, energy, temp_flow
    
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
    
    def cal_cost_df(energy, el_cost=0.4, feedin_tariff=0.07):
        global dt
        energy['export_cost'] = energy['Q2grid']*feedin_tariff *dt
        energy['import_cost'] = energy['Qfrom_grid']*el_cost *dt
        energy['el_bill'] = energy['import_cost']-energy['export_cost']
        return energy
    
    def cal_emissions(energy, el_ef=340, gas_ef=2016/1000):
        # returns emissions in kgCO2, 
        # emission factor in gCO2/kWh for electricity, kgCO2/m3 for gas
        global dt
        dt=0.1
        aef = pd.read_csv('AEF_annual.csv', index_col=0)
        aef.columns=aef.columns.map(int)
        energy['aef']=0
        for index, imp in energy['aef'].items():
            energy['aef'].loc[index] = (aef[index.month][index.hour])
            # energy['aef'] = energy['aef']/1000
        energy['emission'] = energy['Qfrom_grid']*energy['aef']*dt/1000
        el_em = ((energy['Qfrom_grid']*energy['aef']).sum())
        gas_em = ((energy['gas']*gas_ef).sum())
        return el_em, gas_em, energy
    
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
    
    def cal_ldc(energy):
        global dt
        energyH = ((energy.resample('H').sum())*dt) [:-1]#hourly energy
        rldc = pd.DataFrame()
        rldc['net_import']=-energyH['Q2grid']+energyH['Qfrom_grid']            
        rldc = rldc.sort_values(by=['net_import'], ascending = False)
        rldc['interval']=1
        rldc['duration'] = rldc['interval'].cumsum()
        rldc['timestamp'] = rldc.index
        rldc.set_index('duration', inplace=True)
        
        ldc = pd.DataFrame()
        ldc['load'] = energyH['Qload']
        ldc = ldc.sort_values(by=['load'], ascending=False)
        ldc['interval'] = 1
        ldc['duration'] = ldc['interval'].cumsum()
        ldc['timestamp'] = ldc.index
        ldc.set_index('duration', inplace=True)
        return rldc,ldc
    
    def cal_opp(rldc):
        import_intercept = rldc[rldc.iloc[:,0] >=0].index[-1] #filter dataframe with first column above 0, and find the last index value for x intercept
        opp_import_intercept = round(0.01*import_intercept)
        opp_import = rldc.loc[opp_import_intercept, rldc.columns[0]]
        
        export_intercept = len(rldc)-import_intercept
        opp_export_intercept = len(rldc) - round(0.01*export_intercept)
        opp_export = rldc.loc[opp_export_intercept, rldc.columns[0]]
        return opp_import, opp_export, opp_import_intercept, opp_export_intercept
        
    def cal_cop(energy):
        COP = {}
        COP['min'] = energy['COP'].min()
        COP['median'] = energy['COP'].median()
        COP['max'] = energy['COP'].max()
        return COP
    
    def unmet_hours(controls,temp_flow):
        """
        Returns
        -------
        temp_flow : dataframe
            calculated unmet hours. At night the acceptable drop is 2 degrees below set point.
            During the day, when the room is occupied, the acceptable drop is 0.85 degrees
            Finally, new columns are created that only record the room temperature, when there 
            are occupants in the room
        """
        global dt
        unmet = []
        for i in temp_flow.index:
            tf = temp_flow.loc[i]
            c = controls.loc[i]
            if ((i.hour<7 and i.hour>22) and (tf['Tfloor1']<(tf['Tset1']-2) or
                                              tf['Tfloor2']<(tf['Tset2']-2))):
                unmet.append(dt)
            elif ((i.hour>=7 and i.hour<=22) and ((tf['Tfloor1']<(tf['Tset1']-1.11) and c['occ_living']>0))):# or
                                                  # (tf['Tfloor2']<(tf['Tset2']-0.85) and c['occ_first']>0))):
                unmet.append(dt)
            else:
                unmet.append(0.0)
        temp_flow['unmet'] = unmet
        
        # occ1 = controls['occ_living'].replace([0,0.5],[np.NaN,1])
        # occ2 = controls['occ_first'].replace([0,0.5],[np.NaN,1])
        # temp_flow['T1_corr'] = temp_flow['Tfloor1']*occ1
        # temp_flow['T2_corr'] = temp_flow['Tfloor2']*occ2
        return temp_flow
    
    def read_results():
        res_folder = 'res\\'
        trn_folder = 'res\\trn\\'
        
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
        dfresults.insert(4,'batt',None)
        dfresults['batt'] = dfresults['design_case'].str.extract(r'(\d+)')
        dfresults['batt'] = dfresults['batt'].fillna(0).astype(int)
        
        return dfresults, rldc
    
    @staticmethod
    def create_dfs(label, file):
        global DT, dt, t_start, t_end

        if 'cp' in label:
            controls, energy, temp_flow = PostprocessFunctions.cal_base_case(file)
            
        else:
            temp_flow = pd.read_csv(file+'_temp_flow.txt', delimiter=",",index_col=0)
            energy = pd.read_csv(file+'_energy.txt', delimiter=",", index_col=0)
            controls = pd.read_csv(file+'_control_signal.txt', delimiter=",",index_col=0)
            
            controls = PostprocessFunctions.modify_df(controls)#, t_start, t_end)
            temp_flow = PostprocessFunctions.modify_df(temp_flow)#, t_start, t_end)
            energy = PostprocessFunctions.modify_df(energy)/3600#, t_start, t_end)/3600     # kJ/hr to kW 
            energy = PostprocessFunctions.cal_energy(energy, controls)
        
        return controls,energy,temp_flow
    
    @staticmethod
    def create_additional_dfs(file):
        mb = pd.read_csv(file+'_mass_balance.txt', delimiter=",",index_col=0)
        mb = PostprocessFunctions.modify_df(mb)
        param = pd.read_csv(file+'_parameters.txt', delimiter=",",index_col=0)
        param = PostprocessFunctions.extract_params(param)
        return mb, param
    
    def unmet_delta(controls, energy, temp_flow):
        temp_flow['unmet_delta'] = 0
        for i in temp_flow.index:
            if temp_flow.loc[i]['unmet'] > 0:
                tset = temp_flow.loc[i]['Tset1']
                tair = temp_flow.loc[i]['Tfloor1']
                temp_flow.loc[i,'unmet_delta'] = tset-tair
                print(tset-tair)
                
    def investment_cost(design_case, vol, coll_area):
        wwhp = 5400
        ashp = 4700
        pv_panels = 500/1.65    #Eur per m2. or 500 for 1 panel of 1.65 m2
        collector = 300         # Eur/m2
        inverter = 400          # cost of 1 inverter
        batt_6 = 7800
        batt_9 = 9600
        tank_150 =  1250
        vol_increments = 0.05
        min_vol = 0.15
        
        tank = tank_150 * 1.1 **((vol-min_vol)/vol_increments)
        
        match design_case:
            case 'ST':
                cost = coll_area*collector + wwhp + tank
            case 'PVT_0':
                cost = coll_area*collector+ inverter + wwhp + tank
            case 'PVT_6':
                cost = coll_area*collector+ inverter + wwhp + tank + batt_6
            case 'PVT_9':
                cost = coll_area*collector+ inverter + wwhp + tank + batt_9
            case 'ASHP':
                cost = coll_area*pv_panels + inverter + ashp + tank
            case 'cp_PV':
                cost = coll_area*pv_panels + inverter + tank
        return cost  
    
    def make_plot(ax, data, plotstyle, linestyle, color, t1=None,t2=None,
                  ylim1=None, ylim2=None, ylabel='', title=None, legend_loc='best',                      
                  xlabel=0, ygrid=None, sharedx=True, legend=True):
        for d,p,l,c in zip(data, plotstyle, linestyle, color):
            if plotstyle=='area':
                data.plot.area(ax=ax,style=l,color=c, alpha=0.4)
            else:
                data.plot(ax=ax,style=l,color=c)
        PostprocessFunctions.plot_specs(ax,t1,t2,ylim1,ylim2, ylabel,sharedx, legend, legend_loc, title,xlabel,ygrid)
        return ax
        
    def comp_en_for_one_day(energy,day):
        energy_daily,energy_hourly = PostprocessFunctions.cal_daily_hourly(energy)
        en = energy_daily.loc[day]
        return en
    
    def make_sankey_flows(en, parameters, return_extra_data=True):
        if parameters['operation_mode'].iloc[0] != 4:
            en['Qssbuff_load'] = 0
            
            # Define the sources, targets, and labels as a dictionary
            flows = {
                'Irradiance': {'source': 'Irradiance', 'target':'Collector','value': en['QuColl_irr_pos']},
                'Temp diff': {'source': 'Temp diff', 'target':'Collector','value': en['QuColl_t_pos']},
                'coll_loss': {'source': 'Collector', 'target':'Coll_loss','value': -en['QuColl_neg']},
                'HX': {'source': 'Collector', 'target': 'node', 'value': en['Qhx_pos']}, 
                'Coll_load': {'source': 'Collector','target':'HPsource', 'value':(en['QuColl_irr_pos']+en['QuColl_t_pos']-en['Qhx_pos']+en['QuColl_neg'])},
                'Qhp_source': {'source': 'HPsource', 'target': 'HP', 'value': en['Qhp_source']},
                'Qhp': {'source': 'Power', 'target': 'HP', 'value': en['Qhp']},
                'Qhp_load': {'source': 'HP', 'target': 'node', 'value': en['Qhp_load']},
                'Qaux_hp': {'source': 'Aux_hp', 'target': 'node', 'value': en['Qaux_hp']},
                'Qsh_buff_source': {'source': 'node', 'target': 'SH', 'value': -en['Qsh_buff_source']},
                'Qsh_buff_load': {'source': 'SH', 'target': 'SH_load', 'value': en['Qsh_buff_load']},
                'Qloss_sh': {'source': 'SH', 'target': 'SH_loss', 'value': en['Qloss_sh']},
                'Qstored_sh': {'source': 'SH', 'target': 'SH_stored', 'value': en['Qstored_sh']},
                'Qdhw_source': {'source': 'node', 'target': 'DHW', 'value': -en['Qdhw_in']},
                'Qdhw_load': {'source': 'DHW', 'target': 'Tap', 'value': en['Qdhw_load']},
                'Qloss_dhw': {'source': 'DHW', 'target': 'DHW_loss', 'value': en['Qloss_dhw']},
                'Qstored_dhw': {'source': 'DHW', 'target': 'DHW_stored', 'value': en['Qstored_dhw']},
                'HX_loss': {'source':'DHW', 'target':'HX loss', 'value': en['Qdhw_out']}
            }
        else:
            # Define the sources, targets, and labels as a dictionary
            flows = {
                'Irradiance': {'source': 'Irradiance', 'target':'Collector','value': en['QuColl_irr_pos']},
                'Temp diff': {'source': 'Temp diff', 'target':'Collector','value': en['QuColl_t_pos']},
                'coll_loss': {'source': 'Collector', 'target':'Coll_loss','value': -en['QuColl_neg']},
                'HX': {'source': 'Collector', 'target': 'node', 'value': en['Qhx_pos']}, 
                'Qssbuff_source': {'source': 'Collector', 'target': 'SS_buff', 'value': -en['Qssbuff_source']},
                'Qloss_ssbuff': {'source': 'SS_buff', 'target': 'SS_buff_loss', 'value': en['Qloss_ssbuff']},
                'Qstored_buff': {'source': 'SS_buff', 'target': 'SS_buff_stored', 'value': en['Qstored_buff']},
                'Qssbuff_load': {'source': 'SS_buff', 'target': 'HPsource', 'value': en['Qssbuff_load']},
                'Qhp_source': {'source': 'HPsource', 'target': 'HP', 'value': en['Qhp_source']},
                'Qhp': {'source': 'Power', 'target': 'HP', 'value': en['Qhp']},
                'Qhp_load': {'source': 'HP', 'target': 'node', 'value': en['Qhp_load']},
                'Qaux_hp': {'source': 'Aux_hp', 'target': 'node', 'value': en['Qaux_hp']},
                'Qsh_buff_source': {'source': 'node', 'target': 'SH', 'value': -en['Qsh_buff_source']},
                'Qsh_buff_load': {'source': 'SH', 'target': 'SH_load', 'value': en['Qsh_buff_load']},
                'Qloss_sh': {'source': 'SH', 'target': 'SH_loss', 'value': en['Qloss_sh']},
                'Qstored_sh': {'source': 'SH', 'target': 'SH_stored', 'value': en['Qstored_sh']},
                'Qdhw_source': {'source': 'node', 'target': 'DHW', 'value': -en['Qdhw_in']},
                'Qdhw_load': {'source': 'DHW', 'target': 'Tap', 'value': en['Qdhw_load']},
                'Qloss_dhw': {'source': 'DHW', 'target': 'DHW_loss', 'value': en['Qloss_dhw']},
                'Qstored_dhw': {'source': 'DHW', 'target': 'DHW_stored', 'value': en['Qstored_dhw']},
                'HX_loss': {'source':'DHW', 'target':'HX loss', 'value': en['Qdhw_out']}
            }
        # Extract unique labels
        unique_labels = list(set([flow['source'] for flow in flows.values()] + [flow['target'] for flow in flows.values()]))
        unique_labels = sorted(unique_labels, key=lambda x: (x != 'SH', x == 'DHW', x))

        # Create a mapping from labels to indices
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        # Update flows with indices
        for key in flows:
            if flows[key]['value'] < 0 and key not in ['Qssbuff_source', 'Qdhw_source', 'Qsh_buff_source', 'QuColl_neg']:
                flows[key]['value'] *= -1  # Make the value positive
                flows[key]['source'], flows[key]['target'] = flows[key]['target'], flows[key]['source']  # Swap source and target
            flows[key]['source'] = label_to_index[flows[key]['source']]
            flows[key]['target'] = label_to_index[flows[key]['target']]

        if return_extra_data:        
            return flows, unique_labels, label_to_index
        else:
            return flows
        
    def sankey_node_colors(unique_labels, sources):
        node_colors_dict = {'Irradiance': 'rgba(255, 223, 0, 0.8)',
                            'Temp diff': 'rgba(240,90,56,0.8)',
                            'Collector': 'rgba(255, 165, 0, 0.8)',
                            'Coll_loss': 'rgba(255, 69, 0, 0.8)',
                            'SS_buff': 'rgba(70, 130, 180, 0.8)',
                            'SS_buff_loss': 'rgba(123, 104, 238, 0.8)',
                            'SS_buff_stored': 'rgba(106, 90, 205, 0.8)',
                            'node': 'rgba(60, 179, 113, 0.8)',
                            'HPsource': 'rgba(138, 43, 226, 0.8)',
                            'HP': 'rgba(75, 0, 130, 0.8)',
                            'Power': 'rgba(0, 0, 255, 0.8)',
                            'Aux_hp': 'rgba(255, 20, 147, 0.8)',
                            'SH': 'rgba(255, 105, 180, 0.8)',
                            'SH_load': 'rgba(255, 182, 193, 0.8)',
                            'SH_loss': 'rgba(255, 140, 0, 0.8)',
                            'SH_stored': 'rgba(250, 159, 133, 0.8)',
                            'DHW': 'rgba(0, 255, 255, 0.8)',
                            'Tap': 'rgba(173, 216, 230, 0.8)',
                            'DHW_loss': 'rgba(32, 178, 170, 0.8)',
                            'DHW_stored': 'rgba(0, 206, 209, 0.8)',
                            'HX loss': 'black',
                            }

        # Set node colors and link colors
        node_colors = [node_colors_dict[label] for label in unique_labels]
        link_colors = [node_colors_dict[unique_labels[src]].replace('0.8', '0.3') for src in sources]
        return node_colors, link_colors
    
    def sankey_node_positions(unique_labels):
        node_positions = {'Irradiance': (0.1, 0.2),
                          'Temp diff':(0.1, 0.4),
                          'Collector': (0.2, 0.3),
                          'Coll_loss': (0.3, 0.15),
                          'SS_buff': (0.4, 0.5),
                          'SS_buff_loss': (0.5, 0.1),
                          'SS_buff_stored': (0.5, 0.25),
                          'HPsource': (0.45, 0.3),
                          'HP': (0.55, 0.38),
                          'Power': (0.1, 0.62),
                          'Aux_hp': (0.1, 0.8),
                          'node': (0.65, 0.38),
                          'SH': (0.8, 0.2),  # SH at the top
                          'SH_load': (0.9, 0.1),
                          'SH_loss': (0.9, 0.25),
                          'SH_stored': (0.75, 0.3),
                          'DHW': (0.75, 0.55),  # DHW at the bottom
                          'Tap': (0.9, 0.48),
                          'DHW_loss': (0.9, 0.7),
                          'DHW_stored': (0.7, 0.7),
                          'HX loss': (0.9,0.8), }

        # Get positions for all nodes, defaulting to (0.5, 0.5) if not specified
        node_x = [node_positions[label][0] if label in node_positions else 0.5 for label in unique_labels]
        node_y = [node_positions[label][1] if label in node_positions else 0.5 for label in unique_labels]
        return node_x, node_y
    
    def sankey_node_totals(unique_labels, sources,targets,values):
        node_totals = {label: 0 for label in unique_labels}
        node_totals_load = {label: 0 for label in unique_labels}
        for source, target, value in zip(sources, targets, values):
            node_totals[unique_labels[source]] -= value  # Outgoing value
            node_totals_load[unique_labels[target]] += value  # Incoming value
        return node_totals, node_totals_load
    
    def compare_monthly_bars(m, compare1,compare2):
        """
        inputs: m --> dictionaly containing monthly aggregates
                compare1/compare2 --> names of files to be compared
        """
        barWidth = 0.08
        r1 = np.arange(len(m[compare1]))
        r2 = [x + barWidth for x in r1]
        r3 = [x + 2 * barWidth for x in r1]
        r4 = [x + 3 * barWidth for x in r1]
        r5 = [x + 4 * barWidth for x in r1]
        r6 = [x + 5 * barWidth for x in r1]
        r7 = [x + 6 * barWidth for x in r1]
        r8 = [x + 7 * barWidth for x in r1]
        r9 = [x + 8 * barWidth for x in r1]
        r10 = [x + 9 * barWidth for x in r1]

        plt.figure(figsize=(9, 7))

        # Plotting for test35
        bars1 = plt.bar(r1, m[compare1]['Qhp'], color='tab:blue', width=barWidth, edgecolor='grey', label='Qhp')
        bars2 = plt.bar(r3, m[compare1]['Qaux_hp'], color='tab:orange', width=barWidth, edgecolor='grey', label='Qaux_hp')
        bars3 =  plt.bar(r5, m[compare1]['Qaux_dhw'], color='tab:green', width=barWidth, edgecolor='grey', label='Qaux_dhw')
        bars7 =  plt.bar(r7, m[compare1]['Qheat'], color='tab:red', width=barWidth, edgecolor='grey', label='Qheat')
        # bars9 = plt.bar(r7, m[compare1]['Qloss_load'], color='grey', width=barWidth, edgecolor='grey', bottom=m[compare1]['Qheat'],  label='Qloss_load')

        bars4 = plt.bar(r2, m[compare2]['Qhp'], color='tab:blue', width=barWidth, edgecolor='grey',alpha=0.4, label='Qhp')
        bars5 = plt.bar(r4, m[compare2]['Qaux_hp'], color='tab:orange', width=barWidth, edgecolor='grey',alpha=0.4, label='Qaux_hp')
        bars6 = plt.bar(r6, m[compare2]['Qaux_dhw'], color='tab:green', width=barWidth, edgecolor='grey',alpha=0.4, label='Qaux_dhw')
        bars8 =  plt.bar(r8, m[compare2]['Qheat'], color='tab:red', width=barWidth, edgecolor='grey', alpha=0.4, label='Qheat')
        # bars10 = plt.bar(r8, m[compare2]['Qloss_load'], color='grey', width=barWidth, edgecolor='grey', alpha=0.4, bottom=m[compare2]['Qheat'], label='Qloss_load')

        plt.xlabel('Month', fontweight='bold')
        month = m[compare1].index
        plt.xticks([r + 2.5 * barWidth for r in range(len(m[compare1]['Qhp']))], month)

        legend1 = plt.legend([bars1, bars2, bars3, bars7], ['Qhp', 'Qaux_hp', 'Qaux_dhw','Qheat'], loc='upper left')
        # legend2 = plt.legend([bars1, bars4], ['without buffer', 'with buffer'], loc='upper right')
        legend2 = plt.legend([bars1, bars4], [compare1, compare2], loc='upper right')

        plt.gca().add_artist(legend1)

        plt.title('Monthly Energy Consumption Comparison')
        plt.ylabel('Energy consumption [kWh]')
    
    
    def check_daily_totals(t1,t2,energy,keys):
        # keys should a list!
        e = energy.loc[t1:t2]
        total = []
        for key in keys:
            value = round(e[key].sum()*0.1,2)
            total.append(value)
            print(f"{key} = {value} kWh")
            
    def plot_energy_cumulatives(compare, e, which):
        c_yearly, c_monthly, c_daily = {},{},{}
        for label in compare:
            cumulative_yearly, cumulative_monthly, cumulative_daily = PostprocessFunctions.cal_cumulative(e[label])
            c_yearly[label] = cumulative_yearly
            c_monthly[label] = cumulative_monthly
            c_daily[label] = cumulative_daily

        df = c_daily if which=='daily' else c_monthly if which=='monthly' else c_yearly
        styles = ['-', '--', '-.', ':']
        style_handles = []
        lines1, lines2 = [], []
        lines3 = []
        fig_cum, (ax_cum1,ax_cum2,ax_cum3) = plt.subplots(3,1, figsize=(10,7))

        for i,label in enumerate(compare):
            lines1.append(df[label][['Qhp','Qaux_hp']].plot(ax=ax_cum1, color=['blue', 'black'], style=styles[i]))
            lines2.append(df[label]['Qheat'].plot(ax=ax_cum2, color='red', style=styles[i]))
            lines3.append(df[label]['emission'].plot(ax=ax_cum3, color='green', style=styles[i]))
            style_handles.append(plt.Line2D([0], [1], linestyle=styles[i], color='black', label=label))
            
        color_handles1 = [plt.Line2D([0], [0], color='blue', label='Qhp'),
                         plt.Line2D([0], [0], color='black', label='Qaux_hp')]

        color_handles2 = [plt.Line2D([0], [0], color='red', label='Qheat')]

        color_handles3 = [plt.Line2D([0], [0], color='green', label='Emissions')]

        color_legend_ax1 = ax_cum1.legend(handles=color_handles1, loc='upper left')
        style_legend_ax1 = ax_cum1.legend(handles=style_handles, loc='upper right')
        ax_cum1.add_artist(color_legend_ax1)
        ax_cum1.add_artist(style_legend_ax1)

        color_legend_ax2 = ax_cum2.legend(handles=color_handles2, loc='upper left', frameon=False)
        color_legend_ax3 = ax_cum3.legend(handles=color_handles3, loc='upper left', frameon=False)

        ax_cum1.set_ylabel('Qhp and Qaux_hp')
        ax_cum2.set_ylabel('Qheat [kWh]')
        ax_cum3.set_ylabel('emissions [kgCO2]')
        plt.show()
        
    def plot_kpi_cumulatives(compare, e, which):
        c_yearly, c_monthly, c_daily = {},{},{}
        for label in compare:
            cumulative_yearly, cumulative_monthly, cumulative_daily = PostprocessFunctions.cumulative_emission_costs(e[label])
            c_yearly[label] = cumulative_yearly
            c_monthly[label] = cumulative_monthly
            c_daily[label] = cumulative_daily

        df = c_daily if which=='daily' else c_monthly if which=='monthly' else c_yearly
        styles = ['-', '--', '-.', ':']
        style_handles = []
        lines1, lines2 = [],[]
        fig_cum, (ax_cum1, ax_cum2) = plt.subplots(2,1, figsize=(10,7))

        for i,label in enumerate(compare):
            lines1.append(df[label]['emission'].plot(ax=ax_cum1, color='green', style=styles[i]))
            lines2.append(df[label]['el_bill'].plot(ax=ax_cum2, color= 'black', style=styles[i]))
            
            style_handles.append(plt.Line2D([0], [1], linestyle=styles[i], color='black', label=label))
            
        color_handles1 = [plt.Line2D([0], [0], color='green', label='emissions')]
        color_handles2 = [plt.Line2D([0], [0], color='black', label='El bill')]


        color_legend_ax1 = ax_cum1.legend(handles=color_handles1, loc='upper left')
        style_legend_ax1 = ax_cum1.legend(handles=style_handles, loc='upper right')

        color_legend_ax2 = ax_cum2.legend(handles=color_handles2, loc='upper left')
        style_legend_ax2 = ax_cum2.legend(handles=style_handles, loc='upper right')
        ax_cum1.add_artist(color_legend_ax1)
        ax_cum2.add_artist(color_legend_ax2)

        ax_cum1.set_ylabel('Emission [kgCO2]')
        ax_cum2.set_ylabel('Electricity bill [EUR]')

        plt.show()
        
    def new_columns_for_map(temp_flow, controls):
        temp_flow['thp_load_in_dhw'] = temp_flow['Thp_load_in']*(controls['ctr_dhw'])
        controls['no_dhw'] = controls['ctr_dhw'].apply(lambda x: 1 if x == 0 else 0)
        temp_flow['thp_load_in_sh'] = temp_flow['Thp_load_in']*(controls['no_dhw'])
        temp_flow['thp_load_in_dhw'].replace(0,np.NaN, inplace=True)
        temp_flow['thp_load_in_sh'].replace(0,np.NaN, inplace=True)
        return temp_flow, controls