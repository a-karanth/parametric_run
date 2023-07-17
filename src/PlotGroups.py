# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:47:28 2023

@author: 20181270
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from Plots import Plots
from PostprocessFunctions import PostprocessFunctions as pf
pd.options.mode.chained_assignment = None  
matplotlib.rcParams['lines.linewidth'] = 0.8
matplotlib.rcParams["figure.autolayout"] = True

class PlotGroups:
    def __init__(self, con,en,tf, monthly, annual, prefix, label, colors,alpha):
        # importing dictionaries with groups of data
        self.con=con
        self.en=en
        self.tf = tf
        self.monthly = monthly
        self.annual = annual
        self.prefix = prefix
        self.label = label
        self.colors = colors
        self.alpha = alpha
    
    def plot_stacked(self):
        fig, ax = plt.subplots(figsize=(4,5))

        for prefix, label in zip(self.prefix, self.label):            
            x = self.annual[prefix][['Qltg','Qdev','Qhp4sh','Qhp4tank','Qaux_dhw']] # all
            # x = self.annual[prefix][['Qhp4sh','Qhp4tank','Qaux_dhw']] # heating
            # x = self.annual[prefix][['Qltg','Qdev']] # electricty
            x.index = [label]
            bottom = 0
            
            color = ['mediumturquoise','slategrey', 'indianred','orange', 'bisque'] # all
            # color = ['indianred','orange', 'bisque'] # heating
            # color = ['mediumturquoise','slategrey'] # electricity
            for i,c in zip(x,color):
                ax.bar(x[i].index, x[i][0], bottom=bottom, color=c, align='edge')
                if x[i][0] >0:
                    ax.text(x[i].index, x[i][0]+bottom-140, round(x[i][0]), fontsize=9)
                else:
                    ax.text(x[i].index, x[i][0]+bottom-140, "" )
                bottom+=x[i][0]
        ax.grid(axis='y', linestyle='--', alpha=0.4, lw=0.6)
        ax.legend(['Lighting','Devices','SH','DHW','DHW_Aux'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, 
                  shadow=True, ncol=3) #all
        # ax.legend(['SH','DHW','DHW_Aux'],
        #           loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, 
        #           shadow=True, ncol=3) # heating
        # ax.legend(['Lighting','Devices'],
        #           loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, 
        #           shadow=True, ncol=3) # electricity
        pf.plot_specs(ax,None,None,None,None,ylabel='Annual energy consumption [kWh]', title='Annual energy consumption ')
        # ax.set_ylabel('Annual energy consumption [kWh]')
        # ax.set_title('Annual energy consumption ')
        return fig,ax

    
    def plot_ldc(self):
        fig,ax = plt.subplots(figsize=(7,4))
        ldc_out,rldc_out = {},{}
        for prefix,label,color in zip(self.prefix,self.label,self.colors):
            energy = self.en[prefix]
            energyH = (energy.resample('H').sum())*0.1
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
            
            ax.plot(rldc['duration'],rldc['net_import'], label=label, color=color)
            # ax.plot(ldc['duration'],ldc['load'], color=color, linestyle='--', alpha=0.7)
            
            ldc_out[prefix] = ldc
            rldc_out[prefix] = rldc
        
        pf.plot_specs(ax,0,8760, None,None, ylabel='Net import [kWh]', title='Load duration curve ', ygrid='both',
                      legend_loc='upper right')

        return fig, ax, ldc_out, rldc_out
    
    def plot_monthly(self, design_case):
        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,9),sharex=True)
        x = np.arange(12)
        width = 0.9/len(self.prefix)
        n=-(round(len(self.prefix)/2))*width

        for p,l,c,a in zip(self.prefix, self.label, self.colors, self.alpha):
            ax1.bar(x+n, self.monthly[p]['Q2grid'],width, label=l, color=c, alpha=a)    
            ax2.bar(x+n, self.monthly[p]['Qfrom_grid'],width, label=l, color=c, alpha=a)
            ax3.bar(x+n, self.monthly[p]['Qhp'],width, label=l, color=c, alpha=a)
            n = n+width  
            ax3.set_xticks(x, np.arange(1,13))
            
        pf.plot_specs(ax1, None, None, None, None , title='Electricity export [kWh]', ygrid=True, legend=False)
        pf.plot_specs(ax2, None, None, None, None , title='Electicity import [kWh]', ygrid=True, legend=False)
        pf.plot_specs(ax3, None, None, None, None , title='Heat pump load [kWh]', ygrid=True, legend=False)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines[0:len(self.prefix)], labels[0:len(self.prefix)], loc='upper right', ncol=len(design_case))
            