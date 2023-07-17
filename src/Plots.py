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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 0.8
matplotlib.rcParams["figure.autolayout"] = True
import matplotlib.dates as mdates
from PostprocessFunctions import PostprocessFunctions as pf

class Plots:
    def __init__(self, con,en,tf):
        self.temp_flow = tf
        self.energy = en
        self.controls = con
        
    def select_q(self, ax, t1, t2, *args):        
        for i in args:
            self.energy[i].plot(ax=ax)
        pf.plot_specs(ax, t1,t2, None, None, ylabel='p')
            
    def plot_hp(self, ax, t1, t2):
        self.energy[['Qhp4dhw','Qhp4irr']].plot.area(ax=ax, alpha=0.4)
        self.energy['Qhp'].plot(ax=ax, color='black', linewidth=0.6, alpha=0.6)
        pf.plot_specs(ax, t1, t2, None, None, title='Heat pump load [kW]', ygrid=True)
        
    def plot_dhw(self, ax, t1, t2):
        self.temp_flow[['T1_dhw','Tavg_dhw']].plot(ax=ax)
        pf.plot_specs(ax, t1, t2, None, None, ylabel='t', title='DHW Tank', ygrid=True)
    
    def plot_dhw_aux(self, ax, t1, t2):
        self.energy['Qaux_dhw'].plot.area(alpha=0.4)
        pf.plot_specs(ax, t1, t2, None, None, ylabel='Aux demand [kW]', ygrid=True)
        
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
        
        
    def plot_wea(self, temp_flow, energy, month):
        t1 = datetime(2001, month, 14, 0,0,0)
        t1 = t1 + relativedelta(day=1)
        t2 = t1 + relativedelta(day=31)+timedelta(days=1)
        fig, ax = plt.subplots(figsize = (10,6))
        self.temp_flow['Tamb'].plot(ax=ax, color = 'darkred')
        pf.plot_specs(ax, t1,t2,-10,40, ylabel='t', title='Weather for month '+str(month),
                      legend_loc='upper left')
        ax0 = ax.twinx()
        self.energy['Qirr'].plot(ax=ax0, color = 'orange')
        pf.plot_specs(ax0, t1,t2,0,1.2, ylabel='irr', legend_loc='upper right', ygrid=True)
        
    def plot_controls(self,t1, t2):
        ctr, (wea,hp,cdhw,croom,csh) = plt.subplots(5,1, figsize=(19,9),sharex=True)
        self.temp_flow['Tamb'].plot(ax=wea)
        wea0 = wea.twinx()
        self.energy['Qirr'].plot(ax=wea0, color='gold')
        pf.plot_specs(wea, t1,t2, None,None, ylabel='t',title='Weather')
        pf.plot_specs(wea0,t1, t2, ylabel='irr',legend_loc='lower right', ygrid=False)
        
        self.controls['ctr_hp'].plot.area(ax=hp, alpha=0.4, color='orange')
        self.controls['hp_div'].plot(ax=hp, style='--', color='black')
        pf.plot_specs(hp, t1,t2,0,1.1, title='Heat pump signals', legend_loc='lower right')
        
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
        
        self.temp_flow[['Tfloor1','Tfloor2']].plot(ax=croom, color=['red','green'])
        self.temp_flow[['Tset1','Tset2']].plot(ax=croom, color=['red','green'], alpha=0.5)
        croom0 = croom.twinx()
        self.controls['heatingctr1'].plot.area(ax=croom0, alpha=0.4, color='orange')
        self.controls['heatingctr2'].plot(ax=croom0, color='black', style='--')
        pf.plot_specs(croom, t1,t2,8,None, ylabel='Room temperature [deg C]', 
                      legend_loc='lower left', ygrid=True)
        pf.plot_specs(croom0, t1,t2,0,1.1, title='Thermostat signals', legend_loc='lower right')
        
        self.controls['ctr_sh'].plot.area(ax=csh, alpha=0.4, color='orange')
        # controls['sh_pump'].plot(ax=csh)
        self.controls['sh_div'].plot(ax=csh,style='--', color='black')
        pf.plot_specs(csh, t1,t2,0,1.1, title='SH loop signals')
        
    def plot_sh(self,t1,t2):
        fig, (csh, c, t, f, tank) = plt.subplots(5,1, figsize=(19,9),sharex=True)
        self.controls['ctr_sh'].plot.area(ax=csh, alpha=0.4, color='orange')
        self.controls['sh_div'].plot(ax=csh,style='--', color='black')
        self.controls[['ctr_dhw','ctr_irr']].plot(ax=csh, style='-^')
        pf.plot_specs(csh, t1,t2,0,1.1, title='SH loop signals')

        self.controls['heatingctr1'].plot.area(ax=c, alpha=0.4, color='orange')
        self.controls['heatingctr2'].plot(ax=c, color='black', style='--')
        
        self.temp_flow[['Tfloor1','Tfloor2']].plot(ax=t, color=['red','green'])
        self.temp_flow[['Tset1','Tset2']].plot(ax=t, color=['red','green'], alpha=0.5)       
        
        self.temp_flow['mrad1_in'].plot.area(ax=f, alpha=0.3)
        self.temp_flow['mrad2_in'].plot(ax=f)
        f0 = f.twinx()
        self.energy[['Qrad1','Qrad2']].plot(ax=f0, style='--')
        
        self.temp_flow[['T1_sh','T4_sh','T6_sh']].plot(ax=tank)
        
        pf.plot_specs(csh, t1,t2, title='space heating loop controls')

        pf.plot_specs(csh, t1,t2, title='space heating controls')
        pf.plot_specs(c, t1,t2,0,1.1, title='Thermostat signals', legend_loc='lower right')
        pf.plot_specs(t, t1,t2,8,None, ylabel='Room temperature [deg C]', 
                      legend_loc='lower left', ygrid=True)
        pf.plot_specs(f, t1,t2, title='Flow rate and heat exchanfe from radiators', 
                      ygrid=True, legend_loc='lower left')
        pf.plot_specs(f0, t1,t2, legend_loc='lower right')
        pf.plot_specs(tank, t1,t2, title='tank temperatures', ylabel='t', ygrid=True)

        
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
        
    def plot_colormap(self, ax, column, label, cmap='bone', vmin=None, vmax=None):
        
        DT = (self.energy.index[1]-self.energy.index[0]).seconds/60
        day = self.energy.index.dayofyear
        if column != 'Tavg_dhw':
            data = self.energy[column]
        else:
            data = self.temp_flow[column]
        data = data[:-1]
        data = data.values.reshape(int(60*24/DT), len(day.unique()), order="F")

        xgrid = np.arange(day.max()) 
        ygrid = np.arange(0,24,0.1)

        # fig, ax = plt.subplots()
        heatmap = ax.pcolormesh(xgrid, ygrid, data, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks([0, 31, 59, 90, 120, 151, 181, 212, 242, 273, 303, 334] )
        ax.set_xticklabels([str(mdates.num2date(i).strftime('%b %d')) for i in 
                            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]])
        plt.setp(ax.get_xticklabels(), rotation=0, ha="left", rotation_mode="anchor")

        ax.set_yticks(np.arange(0,24,2), fontsize=10)
        hours = [f'{h:02d}:00' for h in range(0,24,2)]
        ax.set_yticklabels(hours)

        # Add a colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.set_label(label)
        ax.set_frame_on(False) # remove all spines

        ax.set_title(column)
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
        
    