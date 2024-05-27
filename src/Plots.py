# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:44:58 2023

@author: 20181270
"""
import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 0.8
matplotlib.rcParams["figure.autolayout"] = True
import matplotlib.dates as mdates
from PostprocessFunctions import PostprocessFunctions as pf
import seaborn as sns

class Plots:
    def __init__(self, con,en,tf):
        self.temp_flow = tf
        self.energy = en
        self.controls = con
        
    def select_q(self, ax, t1, t2, *args):        
        for i in args:
            self.energy[i].plot(ax=ax)
        pf.plot_specs(ax, t1,t2, None, None, ylabel='p')
            
    def plot_t_coll(self, ax, t1,t2):
        self.temp_flow[['Tcoll_in','Tcoll_out','Tamb']].plot(ax=ax, color=['tab:blue','tab:orange','darkred'])
        if 'Thx_source_out' in self.temp_flow.columns:
            self.temp_flow['Thx_source_out'].plot(ax=ax, color='tab:blue', style='--')
        pf.plot_specs(ax,t1,t2,None,None,ylabel='t', title='Collector panel',
                      legend_loc='upper left', ygrid=True)
        
    def plot_t_coll_ssbuff(self, ax, t1,t2):
        self.temp_flow[['Tcoll_in','Tcoll_out','Tamb']].plot(ax=ax, color=['tab:blue','tab:orange','darkred'])
        if 'Thx_source_out' in self.temp_flow.columns:
            self.temp_flow['Thx_source_out'].plot(ax=ax, color='tab:blue', style='--')
        self.temp_flow[['Tssbuff_load_out','Tssbuff_source_out']].plot(ax=ax, label=['Tss_top','Tss_bottom'], color=['mediumvioletred','palevioletred'])
        pf.plot_specs(ax,t1,t2,None,None,ylabel='t', title='Collector panel',
                      legend_loc='upper left', ygrid=True)
    
    def plot_q_coll(self, ax, t1,t2):
        self.energy['QuColl'].plot.area(ax=ax, color=['gold'],alpha=0.2,stacked=False)
        pf.plot_specs(ax,t1,t2,None,None,ylabel='q', legend_loc='upper right')
        
    def plot_c_coll(self, ax, t1,t2):
        self.controls[['coll_pump','ctr_irr','ctr_coll_t']].plot(ax=ax, style='--', color=['gray','orange', 'red'])
        self.energy['Qirr'].plot(ax=ax,color='orange',linewidth=2)
        pf.plot_specs(ax,t1,t2,None,1.5,ylabel='controls', legend_loc='upper right')
        
    def plot_c_coll_ssbuff(self, ax, t1,t2):
        self.controls[['coll_pump','ctr_irr','ctr_coll_t']].plot(ax=ax, style='--', color=['gray','orange', 'red'])
        self.energy['Qirr'].plot(ax=ax,color='orange',linewidth=2)
        self.controls['hx_bypass'].plot(ax=ax, color='skyblue', alpha=1, marker ='*', markersize=4)
        self.controls['ssbuff_stat'].plot(ax=ax, color='black', marker='s', alpha=0.5, markersize=2)
        pf.plot_specs(ax,t1,t2,None,1.5,ylabel='controls', legend_loc='upper right')
        
    def plot_t_hp(self,ax,t1,t2):
        self.temp_flow[['Tcoll_out','Thp_source_out','Thp_load_out','Thp_load_in']].plot(ax=ax,
                                                                                         color=['olivedrab','yellowgreen',
                                                                                                'firebrick','tab:red'])
        pf.plot_specs(ax,t1,t2,None,100, ylabel='t',title='HP', 
                      legend_loc='upper left',ygrid=True)
        
    def plot_q_hp(self, ax, t1, t2):
        self.energy[['Qhp4dhw','Qhp4irr','Qaux_hp']].plot.area(ax=ax, alpha=0.4,stacked=False)
        self.energy['Qhp'].plot(ax=ax, color='black', style='--')
        
        pf.plot_specs(ax, t1, t2, None, None, legend_loc='upper right')
        
    def plot_c_hp(self, ax, t1, t2):
        self.controls['ctr_hp'].plot(ax=ax,style=':', color='black', alpha=0.8)
        if 'hp_div' in self.controls.columns:
            self.controls['hp_div'].plot(ax=ax,color='black',style='--')
        elif 'div_load' in self.controls.columns:
            self.controls['div_load'].plot(ax=ax,color='black',style='--')
        self.controls['demand'].plot(ax=ax, alpha=0.5,color="skyblue")
        ax.fill_between(self.controls.index, self.controls['demand'], color="skyblue", alpha=0.4, hatch='//')

        pf.plot_specs(ax, t1, t2, None, 1.2, legend_loc='upper right')
        
    def plot_t_dhw(self, ax, t1, t2):
        self.temp_flow['T1_dhw'].plot(ax=ax, color='rebeccapurple', alpha=1.0, label='T1_dhw')
        self.temp_flow['Tavg_dhw'].plot(ax=ax, color='rebeccapurple', alpha=0.6, label='Tavg_dhw')
        self.temp_flow['T6_dhw'].plot(ax=ax, color='rebeccapurple', alpha=0.3, label='T6_dhw')
        self.controls['tset_dhw'].plot(ax=ax, color='rebeccapurple', style='--', label='Tset_dhw')
        pf.plot_specs(ax, t1, t2, None, 85, ylabel='t', title='DHW Tank', 
                      legend_loc='upper left', ygrid=True)
    
    def plot_q_dhw(self, ax, t1, t2):
        self.energy['Qaux_dhw'].plot.area(ax=ax, alpha=0.4,stacked=False)
        pf.plot_specs(ax, t1, t2, None, None, ylabel='Aux demand [kW]', 
                      legend_loc='upper right', ygrid=True)
        
    def plot_t_shbuff(self, ax, t1, t2):
        self.temp_flow[['T1_sh']].plot(ax=ax, color='firebrick', alpha=1.0, label='T1_sh')
        self.temp_flow[['Tavg_sh']].plot(ax=ax, color='firebrick', alpha=0.6, label='Tavg_sh')
        self.temp_flow[['T6_sh']].plot(ax=ax, color='firebrick', alpha=0.3, label='T6_sh')
        pf.plot_specs(ax, t1, t2, None, None, ylabel='t', title='DHW Tank', 
                      legend_loc='upper left', ygrid=True)
    
    def plot_c_shbuff(self,ax,t1,t2):
        if 'ctr_buff' in self.controls.columns:
            self.controls[['ctr_dhw','ctr_buff']].plot(ax=ax,style='--', color=['rebeccapurple','firebrick'])
        elif 'ctr_sh_buff' in self.controls.columns:
            self.controls[['ctr_dhw','ctr_sh_buff']].plot(ax=ax,style='--', color=['rebeccapurple','firebrick'])
        pf.plot_specs(ax, t1, t2, None, 1.2, legend_loc='upper right', title='SH buffer and DHW tank control signals')
        
    def plot_t_sh(self, ax, t1,t2):
        self.temp_flow[['Tfloor1','Tfloor2']].plot(ax=ax,color=['mediumvioletred', 'green'])
        self.temp_flow[['Tset1','Tset2']].plot(ax=ax,style=':',color=['mediumvioletred', 'green'])
        if 'Tfloor1_air' in self.temp_flow.columns:
            self.temp_flow[['Tfloor1_air','Tfloor2_air']].plot(ax=ax,style='--',color=['mediumvioletred', 'green'])
        pf.plot_specs(ax, t1, t2, None, 30, ylabel='Room temp [degC]',
                      title='Space heating', legend_loc='upper left', ygrid=True)
    
    def plot_q_sh(self, ax, t1,t2):
        self.energy[['Qrad1','Qrad2']].plot.area(ax=ax,color=['mediumvioletred', 'green'],alpha=0.2,stacked=False)
        pf.plot_specs(ax, t1, t2, None, None, ylabel='Q radiator [kWh]', legend_loc='upper right')
    
    def plot_c_sh(self, ax, t1,t2):
        self.controls['sh_div'].plot(ax=ax,color='black',style='--')
        self.controls['ctr_sh'].plot(ax=ax,style=':', color='black', alpha=0.8)
        pf.plot_specs(ax, t1, t2, None, 1.2, ylabel='Q radiator [kWh]', legend_loc='upper right')
    
    def check_sim(self, t1,t2,file, ssbuff=False):
        fig, ((ax1,ax5),(ax2,ax6),(ax3,ax7),(ax4,ax8)) = plt.subplots(4,2, figsize=(19,9))
        ax10,ax20, ax30, ax40 = ax1.twinx(), ax2.twinx(), ax3.twinx(), ax4.twinx()
        # ax50,ax60, ax70, ax80 = ax5.twinx(), ax6.twinx(), ax7.twinx(), ax8.twinx()
        self.plot_q_coll(ax10,t1,t2)
        if not ssbuff:
            self.plot_t_coll(ax1,t1,t2)
            self.plot_c_coll(ax5,t1,t2)
        
        else:
            self.plot_t_coll_ssbuff(ax1,t1,t2)
            self.plot_c_coll_ssbuff(ax5,t1,t2)
        
        self.plot_q_hp(ax20,t1,t2)
        self.plot_t_hp(ax2,t1,t2)
        self.plot_c_hp(ax6,t1,t2)
        
        self.plot_q_dhw(ax30,t1,t2)
        self.plot_t_dhw(ax3, t1,t2)
        self.plot_t_shbuff(ax3, t1,t2)
        self.plot_c_shbuff(ax7, t1, t2)
       
        self.plot_q_sh(ax40,t1,t2) 
        self.plot_t_sh(ax4,t1,t2) 
        self.plot_c_sh(ax8,t1,t2)
        fig.suptitle(file)
        
        self.plot_unmet()
        plt.suptitle(file)
        
        tsourcein_low = (self.temp_flow['Tcoll_out']<-25)*self.controls['ctr_hp']
        tloadin_low = (self.temp_flow['Thp_load_in']<10)*self.controls['ctr_hp']
        tsourceout_high = (self.temp_flow['Thp_source_out']>35)*self.controls['ctr_hp']
        tloadout_high = (self.temp_flow['Thp_load_out']>60)*self.controls['ctr_hp']
        total_heat = round(self.energy['Qheat'].sum()*0.1,2)
        print(f'Temperature entering HP source is lower than lower limit for {tsourcein_low.sum()} timesteps')
        print(f'Temperature entering HP load is lower than lower limit for {tloadin_low.sum()} timesteps')
        print(f'Temperature exiting HP source is greater than upper limit for {tsourceout_high.sum()} timesteps')
        print(f'Temperature exiting HP load is greater than upper limit for {tloadout_high.sum()} timesteps')
        print(f'Q heat = {total_heat} kWh')
        
        return fig
        
    def plot_sh_dhw(self, t1,t2):    
        comp, (dhw,sh,rt) = plt.subplots(3,1, figsize=(19,9),sharex=True)
        
        self.temp_flow[['T1_dhw','Tavg_dhw','T6_dhw']].plot(ax=dhw, 
                                                            color=['maroon','indianred', 'lightsalmon'])
        self.temp_flow['Thp_load_out'].plot(ax=dhw, color='grey', style='--')
        # self.temp_flow[['mmixDHWout']].plot.area(ax=dhw, alpha=0.2) # not present in older results
        pf.plot_specs(dhw, t1,t2,0,None, ylabel='t', title='DHW tank')
        
        dhw0 = dhw.twinx()
        self.energy['Qaux_dhw'].plot.area(ax=dhw0, alpha=0.2)
        pf.plot_specs(dhw0, t1,t2,0,None, ylabel='Qaux [kW]')
        
        self.temp_flow['mrad1_in'].plot.area(ax=sh, alpha=0.3, color='orange')
        self.temp_flow['mrad2_in'].plot(ax=sh, color='black', style=':')
        pf.plot_specs(sh, t1,t2,0,None, ylabel='flow rate [kg/hr]', title='SH')
        sh0=sh.twinx()
        self.energy[['Qrad1','Qrad2']].plot(ax=sh0)
        pf.plot_specs(sh0, t1,t2,0,10, ylabel='Q [kW]')
        
        self.temp_flow[['Tfloor1','Tset1','Tfloor2','Tset2','Tamb']].plot(ax=rt,color=['firebrick','lightcoral',
                                                                           'green','mediumseagreen',
                                                                           'rebeccapurple'])
        rt0=rt.twinx()
        self.energy['Qirr'].plot(ax=rt0, color='gold')
        pf.plot_specs(rt, t1,t2,None,30, ylabel='t',title='Room temperatures')
        pf.plot_specs(rt0, t1,t2,0,1, ylabel='irr')
        
    def plot_q(self, t1, t2):
        """
        Plot of Qs: HP, aux, PV, to grid, from grid, to-from battery.
        """
        q, (load,batt,temp) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,2, 1]}, figsize=(11,8))
        
        self.energy[['Qpv', 'Qhp', 'Qaux_dhw']].plot(ax=load)
        pf.plot_specs(load, t1,t2, None, 2.7, ylabel='p', title='Electricity demand and generation')

        self.energy[['Qpv', 'Qload', 'Qfrom_grid','Q2grid','Qbatt_tofrom']].plot(ax=batt)
        pf.plot_specs(batt, t1,t2, -2.1, 2.6, ylabel='p', title='Electricty energy balance')
        batt0 = batt.twinx()
        self.energy['SOC'].plot(ax=batt0)
        pf.plot_specs(batt0, t1,t2, -0.1, 6, ylabel='Fractional state of charge', 
                      legend_loc='lower right', ygrid=True)
        
        self.temp_flow[['T1_dhw','Tavg_dhw']].plot(ax=temp)
        pf.plot_specs(temp, t1,t2, 0, 90, ylabel='t', title='Avg tank temperature')
        
        
    def plot_wea(self, month):
        t1 = datetime(2001, month, 14, 0,0,0)
        t1 = t1 + relativedelta(day=1)
        t2 = t1 + relativedelta(day=31)+timedelta(days=1)
        fig, ax = plt.subplots(figsize = (10,6))
        self.temp_flow['Tamb'].plot(ax=ax, color = 'darkred')
        pf.plot_specs(ax, t1,t2,-10,40, ylabel='t', title='Weather for month '+str(month),
                      legend_loc='upper left',ygrid=True)
        
        
        # daily_temp = self.temp_flow['Tamb'].resample('D').agg(['min', 'max'])
        # day = daily_temp.index
        # ax.bar(day, daily_temp['max'], bottom=daily_temp['min'], color='green', alpha=1, width=0.5, label='Daily Temp Range')
        
        ax0 = ax.twinx()
        self.energy['Qirr'].plot(ax=ax0, color = 'orange')
        pf.plot_specs(ax0, t1,t2,0,1.2, ylabel='irr', legend_loc='upper right')
        
    def plot_controls(self,t1, t2):
        ctr, (wea,hp,cdhw,croom,csh) = plt.subplots(5,1, figsize=(19,9),sharex=True)
        self.temp_flow['Tamb'].plot(ax=wea)
        wea0 = wea.twinx()
        self.energy['Qirr'].plot(ax=wea0, color='gold')
        pf.plot_specs(wea, t1,t2, None,None, ylabel='t',title='Weather')
        pf.plot_specs(wea0,t1, t2, ylabel='irr',legend_loc='lower right')
        
        self.controls['ctr_hp'].plot.area(ax=hp, alpha=0.4, color='orange', stacked=False)
        self.controls['hp_div'].plot(ax=hp, style='--', color='black')
        pf.plot_specs(hp, t1,t2,0,1.1, title='Heat pump signals', legend_loc='lower right')
        
        self.temp_flow['Tavg_dhw'].plot.area(ax=cdhw, alpha=0.4, color='orange')
        self.temp_flow['T1_dhw'].plot(ax=cdhw,color='r')
        cdhw0=cdhw.twinx()
        self.controls['ctr_dhw'].plot.area(ax=cdhw0, color='black', alpha=0.3, stacked=False)
        self.controls['night'].plot(ax=cdhw0, marker="2" )
        self.controls['emergency'].plot(ax=cdhw0, marker="+" )
        self.controls['legionella'].plot(ax=cdhw0, marker="x" )
        self. controls['ctr_irr'].plot(ax=cdhw0, marker="1" )
        # controls[['night_charge','aux_signal']].plot(ax=cdhw)
        pf.plot_specs(cdhw, t1,t2,0,85,ylabel='Room temperature [deg C]', 
                      title='DHW loop signals', legend_loc='center left')
        pf.plot_specs(cdhw0, t1,t2,0,None, legend_loc='lower right')
        
        self.temp_flow[['T1_dhw','T6_dhw', 'Tcoll_out']].plot(ax=croom)
        pf.plot_specs(croom, t1,t2,None,None, ylabel='Temperature [deg C]', 
                      legend_loc='lower left')
        croom0 = croom.twinx()
        self.energy['QuColl'].plot.area(croom0, alpha=0.4, stacked=False)
        pf.plot_specs(croom0, t1,t2,-1,None, ylabel='QuColl', 
                      legend_loc='lower right')
        
        # self.temp_flow[['Tfloor1','Tfloor2']].plot(ax=croom, color=['red','green'])
        # self.temp_flow[['Tset1','Tset2']].plot(ax=croom, color=['red','green'], alpha=0.5)
        # croom0 = croom.twinx()
        # self.controls['heatingctr1'].plot.area(ax=croom0, alpha=0.4, color='orange')
        # self.controls['heatingctr2'].plot(ax=croom0, color='black', style='--')
        # pf.plot_specs(croom, t1,t2,8,None, ylabel='Room temperature [deg C]', 
        #               legend_loc='lower left', ygrid=True)
        # pf.plot_specs(croom0, t1,t2,0,1.1, title='Thermostat signals', legend_loc='lower right')
        
        self.controls['ctr_sh'].plot.area(ax=csh, alpha=0.4, color='orange', stacked=False)
        # controls['sh_pump'].plot(ax=csh)
        self.controls['sh_div'].plot(ax=csh,style='--', color='black')
        pf.plot_specs(csh, t1,t2,None,None, title='SH loop signals')
        
        
    def plot_sh(self,t1,t2):
        fig, (csh,f, ret) = plt.subplots(3,1, figsize=(19,9),sharex=True)
        self.controls['sh_div'].plot(ax=csh,style='--', color='black')

        self.temp_flow[['mrad1_in','mrad2_in']].plot.area(ax=f, alpha=0.1, stacked=True)
        self.temp_flow['mrad_mix_out'].plot(ax=ret,linewidth=2, color='teal')
        
        self.temp_flow[['mrad1_out','mrad2_out']].plot.area(ax=ret, alpha=0.1, stacked=True)
      
        pf.plot_specs(csh, t1,t2, title='SH loop signals and mass balance',ygrid=True)
        pf.plot_specs(f, t1,t2, title='Flow rate to radiators', 
                      ygrid=True, legend_loc='upper left')
        pf.plot_specs(ret, t1,t2, title='Flow rate from radiators', 
                      ygrid=True, legend_loc='upper left')
        return fig, csh, f, ret
    
    def plot_coll_loop_ss_buff(self,t1,t2):
        fig, (csh,f, ret) = plt.subplots(3,1, figsize=(19,9),sharex=True)
        self.controls['coll_pump'].plot.area(ax=csh, color='skyblue', alpha=0.1)
        self.controls['hx_bypass'].plot(ax=csh,style='--', color='black')
        
        self.temp_flow['mcoll_in'].plot(ax=f,linewidth=2, color='black', marker='^')
        # self.temp_flow['mcoll_out'].plot(ax=f,linewidth=2, color='teal')
        # self.temp_flow[['mhx_source_in','mssbuff_source_in']].plot.area(ax=f, alpha=0.1, stacked=True)
        
        # self.temp_flow[['mhx_source_out','mssbuff_source']].plot.area(ax=ret, alpha=0.1, stacked=True)
        self.temp_flow['mmix_pvt_out'].plot(ax=ret,linewidth=2, color='teal')
        
        # self.temp_flow[['mrad1_out','mrad2_out']].plot.area(ax=ret, alpha=0.1, stacked=True)
      
        pf.plot_specs(csh, t1,t2, title='SH loop signals and mass balance',ygrid=True)
        pf.plot_specs(f, t1,t2, title='Flow rate to from diverter', 
                      ygrid=True, legend_loc='upper left')
        pf.plot_specs(ret, t1,t2, title='Flow rate to and out of mixer', 
                      ygrid=True, legend_loc='upper left')
        return fig, csh, f, ret

        
    def plot_batt(self, prefix, t_start, t_end):
        q, (ax1,ax2,ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,2, 1]}, 
                                        figsize=(11,8), sharex=True)
        q.suptitle(prefix)
        self.select_q(ax2, t_start,t_end, 'Q2grid', 'Qbatt_tofrom','Qfrom_grid')
        self.energy['Q2load'].plot.area(ax=ax2, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([-1, None])
        # pt.select_q(ax1, t_start,t_end, 'SOC')
        self.select_q(ax1, t_start, t_end, 'Qirr','Qpv')
        ax10 = ax1.twinx()
        self.energy['SOC'].plot.area(ax=ax10, alpha = 0.3)
        ax1.legend()
        ax10.legend()
        self.energy[['Qhp','Qdev','Qltg','Qaux_dhw']].plot(ax=ax3,alpha=0.8)
        pf.plot_specs(ax1,t_start,t_end,-0.1, None,ylabel='irr')
        pf.plot_specs(ax10,t_start,t_end,-0.1, 3,legend_loc='center right')

        
    def plot_summer_loop(self, t1,t2):
        q, (pvt,hp,dhw,ctr) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2,2,2,1]}, figsize=(19,9))
        self.temp_flow[['mhp_source_out','mmixPVTin2']].plot(ax=pvt)
        self.temp_flow['mmixPVTout'].plot.area(ax=pvt, color='green', alpha=0.4)

        self.temp_flow['mhp_load_out'].plot.area(ax=hp, color='green', alpha=0.4)
        self.temp_flow[['msh_in','mhp2dhw','mmixDHWin2' ]].plot(ax=hp)

        self.temp_flow['mdhw_cold_out'].plot.area(ax=dhw, color='green', alpha=0.4)
        self.temp_flow[['mmixPHloadin2','mmixPVTin2']].plot(ax=dhw)

        self.controls['pvt_load_loop'].plot(ax=ctr)
        ctr0 = ctr.twinx()
        self.temp_flow['Tcoll_out'].plot(ax=ctr0,color='red')

        pf.plot_specs(pvt,t1,t2,None,110,title='PVT',ylabel='f')
        pf.plot_specs(hp,t1,t2,None,110,title='HP',ylabel='f')
        pf.plot_specs(dhw,t1,t2,None,110,title='DHW',ylabel='f')
        pf.plot_specs(ctr,t1,t2,None,None,title='ctr',ylabel='control signal')
        pf.plot_specs(ctr0,t1,t2,None,None,ylabel='t')
        
    def plot_colormap(self, ax, df, label, cmap='bone', vmin=None, vmax=None):
        
        DT = (self.energy.index[1]-self.energy.index[0]).seconds/60
        day = self.energy.index.dayofyear
        
        data = df
        data = data[:-1]
        data = data.values.reshape(int(60*24/DT), len(day.unique()), order="F")

        xgrid = np.arange(day.max()) 
        ygrid = np.arange(0,24,0.1)

        # fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(xgrid, ygrid, data, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 242, 273, 303, 334] )
        ax.set_xticklabels([str(mdates.num2date(i).strftime('%b')) for i in 
                            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]])
        plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")

        ax.set_yticks(np.arange(0,24,2))
        hours = [f'{h:02d}:00' for h in range(0,24,2)]
        ax.set_yticklabels(hours, fontsize=10)

        # Add a colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.set_label(label)
        ax.set_frame_on(False) # remove all spines

        ax.set_title(label)
        ax.set_ylabel('Hour of day')
    
    def plot_controls_cp(self,t1, t2):
        ctr, (wea,cdhw,q,croom,csh) = plt.subplots(5,1, figsize=(19,9),sharex=True)
        self.temp_flow['Tamb'].plot(ax=wea)
        wea0 = wea.twinx()
        self.energy['Qirr'].plot(ax=wea0, color='gold')
        pf.plot_specs(wea, t1,t2, None,None, ylabel='t',title='Weather')
        pf.plot_specs(wea0,t1, t2, ylabel='irr',legend_loc='lower right', ygrid=False)
        
        self.temp_flow['Tavg_dhw'].plot.area(ax=cdhw, alpha=0.4, color='orange')
        self.temp_flow['T1_dhw'].plot(ax=cdhw,color='r')
        cdhw0=cdhw.twinx()
        self.controls['ctr_dhw'].plot.area(ax=cdhw0, color='black', alpha=0.3)
        self.controls['night'].plot(ax=cdhw0, marker="2" )
        self.controls['emergency'].plot(ax=cdhw0, marker="+" )
        self.controls['legionella'].plot(ax=cdhw0, marker="x" )
        self. controls['ctr_irr'].plot(ax=cdhw0, marker="1" )
        # controls[['night_charge','aux_signal']].plot(ax=cdhw)
        pf.plot_specs(cdhw, t1,t2,0,85,ylabel='Room temperature [deg C]', 
                      title='DHW loop signals', ygrid=True)
        pf.plot_specs(cdhw0, t1,t2,0,None, legend_loc='lower right')
        
        self.energy[['Qheat_living1','Qheat_living2']].plot(ax=q, color=['orange', 'green'])
        pf.plot_specs(q, t1,t2,-1,None,ylabel='Qheat [kW]', title='Q heat [kW]', legend_loc='lower right', ygrid=True)
        
        self.temp_flow[['Tfloor1','Tfloor2']].plot(ax=croom, color=['red','green'])
        self.temp_flow[['Tset1','Tset2']].plot(ax=croom, color=['red','green'], alpha=0.5)
        croom0 = croom.twinx()
        self.controls['heatingctr1'].plot.area(ax=croom0, alpha=0.4, color='orange')
        self.controls['heatingctr2'].plot(ax=croom0, color='black', style='--')
        pf.plot_specs(croom, t1,t2,8,30, ylabel='Room temperature [deg C]', 
                      legend_loc='lower left', ygrid=True)
        pf.plot_specs(croom0, t1,t2,0,1.1, title='Thermostat signals', legend_loc='lower right', ygrid=True)
        
        self.temp_flow[['Tfloor1','Tfloor1_2']].plot(ax=csh, color=['red','green'])
        pf.plot_specs(csh, t1,t2,8,30, title='Living room temp', ygrid=True)
        
    def scatter_plot(*args, ax, marker, xkey, ykey, ckey, xlabel, ylabel, clabel):
        n = []
        for i in range(len(args)):
            value = (i - (len(args)-1)/2)*0.008
            n.append(value)
        scatter = []
        for df2, m, delta in zip(args, marker, n):
            df = df2.copy()
            df[xkey] = df[xkey] + delta
            scatter.append(ax.scatter(df[xkey], df[ykey],
                                       c=df[ckey], cmap='viridis_r',
                                       marker=m,  alpha=0.7, s=70,
                                       label=df['design_case'].iloc[0]))
        cbar = plt.colorbar(scatter[0], ax=ax)
        cbar.ax.get_yaxis().labelpad = 5
        cbar.ax.set_ylabel(clabel, rotation=90)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7, which='both')
        ax.legend()
        return scatter, cbar
    
    def plot_unmet(self):
        tf = (self.temp_flow.resample('H').mean())[:-1]
        c = (self.controls.resample('H').mean())[:-1]
        df = tf[['Tfloor1','Tfloor2','Tset1','Tset2']]
        df[['occ_living','occ_first']] = c[['occ_living','occ_first']]
        df['T1'] = df['Tfloor1']#*[1 if i>0 else np.NaN for i in df['occ_living']]
        df['T2'] = df['Tfloor2']#*[1 if i>0 else np.NaN for i in df['occ_first']]
        
        df['diff1']= df['T1']-df['Tset1']
        df['diff2']= df['T2']-df['Tset2']
        
        df['low_band1']= [1 if  -0.5<=i<0 else 2 if -1.2<=i<-0.5 else 3 if i<-1.2 else 0 for i in df['diff1']]
        df['low_band2']= [1 if  -0.5<=i<0 else 2 if -1.2<=i<-0.5 else 3 if i<-1.2 else 0 for i in df['diff2']]

        #seaborn plot
        df['day'] = df.index.dayofyear
        df['hour'] = tf.index.hour
        pivot_table1 = df.pivot_table(values='T1', index='hour', columns='day', aggfunc='mean')
        pivot_table2 = df.pivot_table(values='T2', index='hour', columns='day', aggfunc='mean')
        pivot_table3 = df.pivot_table(values='low_band1', index='hour', columns='day', aggfunc='mean')
        pivot_table4 = df.pivot_table(values='low_band2', index='hour', columns='day', aggfunc='mean')
        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(12,8))
        sns.heatmap(pivot_table1,cmap='Spectral_r', ax=ax1,vmin=15,vmax=32, cbar=False)
        sns.heatmap(pivot_table2,cmap='Spectral_r', ax=ax2,vmin=15,vmax=32, cbar=False)
        sns.heatmap(pivot_table3,ax=ax3, cbar=False)
        sns.heatmap(pivot_table4,ax=ax4, cbar=False)
        
        # days_in_year = [0, 31, 59, 90, 120, 151, 181, 212, 242, 273, 303, 334]
        for ax in fig.axes:
            # Set x-axis ticks and labels
            ax.invert_yaxis()
            ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 242, 273, 303, 334] )
            ax.set_xticklabels([str(mdates.num2date(i).strftime('%b')) for i in 
                                [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]])
            plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            # Add color bar
            cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
            if ax==ax3 or ax==ax4:
                cbar.set_ticks([0, 1, 2, 3])
                cbar.set_ticklabels(["0", "0.5", "1.2", ">1.2"])
        
        fig.tight_layout()
        # #matplotlib plot
        # day = tf.index.dayofyear.unique()
        # t1 = df['T1'].values.reshape(24, len(day), order="F")
        # t2 = df['T2'].values.reshape(24, len(day), order="F")
        
        # xgrid = np.arange(day.max()) 
        # ygrid = np.arange(0,24,1)
        
        # fig,(ax1,ax2)=plt.subplots(2,1)
        # heatmap = ax1.pcolormesh(xgrid, ygrid, t1, cmap='Spectral_r', vmin=12, vmax=26)
        # cbar1 = ax1.figure.colorbar(heatmap, ax=ax1)
        # cbar1.set_label('Troom [dgeC]')
        # heatmap = ax2.pcolormesh(xgrid, ygrid, t2, cmap='Spectral_r', vmin=12, vmax=26)
        # cbar2 = ax2.figure.colorbar(heatmap, ax=ax2)
        # cbar2.set_label('Troom [dgeC]')
        
        # ax1.set_ylim([6.5,22.5])
        # ax2.set_ylim([6.5,22.5])
        
        def plot_pvt_load_loop(self, t1,t2):

            fig, (ax2,ax0,ax, ax4) = plt.subplots(4,1)
    
            self.temp_flow[['Tcoll_out','Tcoll_in']].plot(ax=ax2, color=['firebrick','tab:blue'])
            pf.plot_specs(ax2, t1,t2,None,None,ylabel='t', legend_loc='center left', title='Collector panel')
    
            self.temp_flow['mcoll_in'].plot(ax=ax0)
            ax00 = ax0.twinx()
            self.energy['QuColl'].plot(ax=ax00, color='gold')
            pf.plot_specs(ax00, t1,t2,-0.2,None,ylabel='Energy', legend_loc='center right')
    
            self.controls['coll_pump'].plot(ax=ax)
            self.controls['hx_bypass'].plot.area(ax=ax,alpha=0.2)
            pf.plot_specs(ax0, t1,t2,None,None,ylabel='f', legend_loc='center left')
            pf.plot_specs(ax, t1,t2,None,None,ylabel='p', legend_loc='center right')  
    
            self.controls['ctr_irr'].plot.area(ax=ax4, color='gold', alpha=0.2)
            self.controls['ctr_sh'].plot(ax=ax4, marker='*')
            self.controls['ctr_dhw'].plot(ax=ax4, linestyle='--')
            pf.plot_specs(ax4, t1,t2,None,None,ylabel='controls', legend_loc='center right')
    