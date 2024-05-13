# -*- coding: utf-8 -*-
"""

"""

import numpy
import os

thisModule = os.path.splitext(os.path.basename(__file__))[0]
   
def Initialization(TRNData):
    global ctr_hp, div_load, div_pvt, demand, ctrIRR,hp_aux, coll_pump
    global ctr_coll_t, ctrDHW, dhw_stat, hx_bypass, load_hx_bypass
    global tavg_dhw, tavg_buff, tcoll_in, tcoll_out
    global tset_dhw
    global heating_season, ctrSH, sh_div, ctrSHbuff, night, legionella, operation_mode
    # This model has nothing to initialize    
    return

def StartTime(TRNData):

    # Define local short names for convenience (this is optional)
    ctr_hp, div_load, demand, ctrIRR, coll_pump = 0,0,0,0,0
    ctr_coll_t, ctrDHW, dhw_stat, hx_bypass, load_hx_bypass = 0,0,0,0,0
    tavg_dhw, tavg_buff = 30,30
    tset_dhw = 20
    aux_set = 45
    heating_season, ctrSH, sh_div, ctrSHbuff, night, legionella, dhw_demand, operation_mode  =0,0,0,0,0,0,0,0
    
    ctrfloor1 = TRNData[thisModule]["inputs"][0]
    ctrfloor2 = TRNData[thisModule]["inputs"][1]
    dhw_stat = TRNData[thisModule]["inputs"][2]
    hour = TRNData[thisModule]["inputs"][3]
    week = TRNData[thisModule]["inputs"][4]
    month = TRNData[thisModule]["inputs"][5]
    ctrIRR = TRNData[thisModule]["inputs"][6]
    ssbuff_stat = TRNData[thisModule]["inputs"][7]
    dhw_load = TRNData[thisModule]["inputs"][8]
    tavg_buff = TRNData[thisModule]["inputs"][9]
    ctrSHbuff = TRNData[thisModule]["inputs"][10]
    ctr_coll_t = TRNData[thisModule]["inputs"][11]
    tavg_dhw = TRNData[thisModule]["inputs"][12]
    tcoll_in = TRNData[thisModule]["inputs"][13]
    tcoll_out = TRNData[thisModule]["inputs"][14]
    tamb = TRNData[thisModule]["inputs"][15]
    tset_dhw = TRNData[thisModule]["inputs"][16]
    operation_mode = TRNData[thisModule]["inputs"][17]
    
    TRNData[thisModule]["outputs"][0] = ctrfloor1*1
    TRNData[thisModule]["outputs"][1] = ctrfloor2*1
    TRNData[thisModule]["outputs"][2] = heating_season*1
    TRNData[thisModule]["outputs"][3] = ctrSH*1
    TRNData[thisModule]["outputs"][4] = sh_div*1
    TRNData[thisModule]["outputs"][5] = ctrSHbuff*1
    TRNData[thisModule]["outputs"][6] = night*1
    TRNData[thisModule]["outputs"][7] = legionella*1
    TRNData[thisModule]["outputs"][8] = ctrDHW*1
    TRNData[thisModule]["outputs"][9] = demand*1
    TRNData[thisModule]["outputs"][10] = coll_pump*1
    TRNData[thisModule]["outputs"][11] = dhw_demand*1
    TRNData[thisModule]["outputs"][12] = hx_bypass*1
    TRNData[thisModule]["outputs"][13] = load_hx_bypass*1
    TRNData[thisModule]["outputs"][14] = ctr_hp*1
    TRNData[thisModule]["outputs"][15] = div_load*1
    TRNData[thisModule]["outputs"][16] = aux_set*1
    TRNData[thisModule]["outputs"][17] = tset_dhw*1
    # Calculate the outputs
    # Set outputs in TRNData
    return

def cal_controls (control_case, dhw_demand, ctrDHW, ctrSHbuff, hx_bypass, coll_pump, tset_dhw,ctrIRR):
    demand = ctrDHW or ctrSHbuff
    if ctrSHbuff and not(dhw_demand): #if SH and DHW tank have to charge, but there is no dhw demand, then prioritize SH
        div_load = 0
    else:
        div_load = ctrDHW
    
    aux_set = 45 + 5*div_load 
    
    ctr_hp = coll_pump and hx_bypass and demand
    load_hx_bypass = hx_bypass
    
    dhw_set_max = 50
    dhw_set_min = 50
         
    match(control_case):
        case 'base':
            ctr_hp = coll_pump and hx_bypass and demand
            if hx_bypass and not(coll_pump):
                load_hx_bypass=0
            dhw_set_max = 70
            dhw_set_min = 50
            
                
        case 'sahw':
            hx_bypass, load_hx_bypass, ctr_hp = 0,0,0
            dhw_set_max = 60
            dhw_set_min = 50
            
        case 'sdhw':
            hx_bypass, load_hx_bypass, ctr_hp = 0,0,0
            aux_set = 45 - 25*div_load #when ctrDHW, we wnt aux to be off, so setpoint is set at 20
            dhw_set_max = 60
            dhw_set_min = 50
            if coll_pump and ctrSHbuff and not(ctrDHW):
                coll_pump=0
                
    if not(demand) and ctrIRR:
        tset_dhw = dhw_set_max
    elif demand and not(coll_pump):
        tset_dhw = dhw_set_min
    elif tset_dhw <=-9000:
        tset_dhw = 40
    else:
        tset_dhw = tset_dhw
        
        
    return hx_bypass, coll_pump, ctr_hp, demand, load_hx_bypass, div_load, aux_set, tset_dhw

def Iteration(TRNData):

    # Define local short names for convenience (this is optional)
    ctrfloor1 = TRNData[thisModule]["inputs"][0]
    ctrfloor2 = TRNData[thisModule]["inputs"][1]
    dhw_stat = TRNData[thisModule]["inputs"][2]
    hour = TRNData[thisModule]["inputs"][3]
    week = TRNData[thisModule]["inputs"][4]
    month = TRNData[thisModule]["inputs"][5]
    ctrIRR = TRNData[thisModule]["inputs"][6]
    ssbuff_stat = TRNData[thisModule]["inputs"][7]
    dhw_load = TRNData[thisModule]["inputs"][8]
    tavg_buff = TRNData[thisModule]["inputs"][9]
    ctrSHbuff = TRNData[thisModule]["inputs"][10]
    ctr_coll_t = TRNData[thisModule]["inputs"][11]
    tavg_dhw = TRNData[thisModule]["inputs"][12]
    tcoll_in = TRNData[thisModule]["inputs"][13]
    tcoll_out = TRNData[thisModule]["inputs"][14]
    tamb = TRNData[thisModule]["inputs"][15]
    tset_dhw = TRNData[thisModule]["inputs"][16] 
    operation_mode = TRNData[thisModule]["inputs"][17]
    
    heating_season = (month>10 or month<5)
    ctrSH = (ctrfloor1 or ctrfloor2) and heating_season
    sh_div = (ctrfloor1 and ctrfloor2)*0.5 + (ctrfloor1 or ctrfloor2)*(ctrfloor1 < ctrfloor2)
    night = dhw_stat and (5 <hour <=6)
    legionella = (week==7) and (6<hour<=8)

    ctrDHW = (dhw_stat) or legionella or night    
    dhw_demand = dhw_load>0
    
    # coll_pump = ((ctrIRR or ctr_coll_t) or -25<tcoll_in<=tamb)
    coll_pump = ((ctrIRR or ctr_coll_t) or tcoll_in<=tamb) 
    # coll_pump = 0.8 if ctrIRR else 1 if -25<tcoll_in<=tamb or ctr_coll_t else 0
    hx_bypass = tcoll_out<35
    
    control_case = 'base' if operation_mode==1 else 'sahw' if operation_mode==2 else 'sdhw'
    hx_bypass, coll_pump, ctr_hp, demand, load_hx_bypass, div_load, aux_set, tset_dhw = cal_controls(control_case, dhw_demand, 
                                                                                                    ctrDHW, ctrSHbuff, hx_bypass, 
                                                                                                    coll_pump, tset_dhw,ctrIRR)

    # Set outputs in TRNData    
    TRNData[thisModule]["outputs"][0] = ctrfloor1*1
    TRNData[thisModule]["outputs"][1] = ctrfloor2*1
    TRNData[thisModule]["outputs"][2] = heating_season*1
    TRNData[thisModule]["outputs"][3] = ctrSH*1
    TRNData[thisModule]["outputs"][4] = sh_div*1
    TRNData[thisModule]["outputs"][5] = ctrSHbuff*1
    TRNData[thisModule]["outputs"][6] = night*1
    TRNData[thisModule]["outputs"][7] = legionella*1
    TRNData[thisModule]["outputs"][8] = ctrDHW*1
    TRNData[thisModule]["outputs"][9] = demand*1
    TRNData[thisModule]["outputs"][10] = coll_pump*1
    TRNData[thisModule]["outputs"][11] = dhw_demand*1
    TRNData[thisModule]["outputs"][12] = hx_bypass*1
    TRNData[thisModule]["outputs"][13] = load_hx_bypass*1
    TRNData[thisModule]["outputs"][14] = ctr_hp*1
    TRNData[thisModule]["outputs"][15] = div_load*1
    TRNData[thisModule]["outputs"][16] = aux_set*1
    TRNData[thisModule]["outputs"][17] = tset_dhw*1

    return

def EndOfTimeStep(TRNData):
    # print(f"ctrfloor1: {ctrfloor1}")
    # print(f"ctrfloor2: {ctrfloor2}")
    # print(f"heating_season: {heating_season}")
    # print(f"ctrSH: {ctrSH}")
    # print(f"sh_div: {sh_div}")
    # print(f"ctrSHbuff: {ctrSHbuff}")
    # print(f"night: {night}")
    # print(f"legionella: {legionella}")
    # print(f"ctrDHW: {ctrDHW}")
    # print(f"ctrHP: {ctrHP}")
    # print(f"hp_div: {hp_div}")
    # print(f"pvt_load_loop: {pvt_load_loop}")
    # print(f"sh_pump: {sh_pump}")
    # print(f"dhw_set: {dhw_set}")
    # print(f"sh_set: {sh_set}")
    return


def LastCallOfSimulation(TRNData):
    return