# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:06:54 2023

@author: 20181270
"""

class ModifyType56:
    def __init__(self):
        self.brick = 'Brick'
        self.stucco = 'Stucco'
        
        self.air_cav = 'Aircavity'
        self.air_cav_floor = 'Aircavity_floor'
        self.fibreglass = 'Fibreglass'
        self.pu = 'Polyurethane'
        self.roofdeck = 'Roofdeck'
        self.rooftile = 'Rooftile'
        self.plasterboard = 'Plasterboard'
        self.timber = 'Timberfloor'
        self.concrete = 'Concrete8in'
        
    brick = 'Brick '
    stucco = 'Stucco '
    air_cav = 'Aircavity '
    air_cav_floor = 'Aircavity_floor '
    fibglass = 'Fibreglass '
    pu = 'Polyurethane '
    roofdeck = 'Roofdeck '
    rooftile = 'Rooftile '
    plasterboard = 'Plasterboard '
    timber = 'Timberfloor '
    concrete = 'Concrete8in '
    
    cm0 = '0\t'
    cm1 = '0.01\t'
    cm2 = '0.02\t'
    cm3 = '0.03\t'
    cm5 = '0.05\t'
    cm6 = '0.06\t'
    cm7 = '0.07\t'
    cm10 = '0.1\t'
    cm15 = '0.15\t'
    cm20 = '0.2\t'
    cm25 = '0.25\t'
    
    layer = ' LAYERS   = '
    thickness = ' THICKNESS= '
    
    r0_wall_con =  brick + air_cav + brick
    r0_wall_thk = cm10 + cm0 + cm10
    r0_wall = layer + r0_wall_con + '\n' + thickness + r0_wall_thk + '\n'
    r0_roof_con = plasterboard+ air_cav + timber + roofdeck + rooftile
    r0_roof_thk = cm2 + cm0 + cm7 + cm3 + cm3
    r0_roof = layer + r0_roof_con + '\n' + thickness + r0_roof_thk + '\n'
    r0_floor_con = concrete + air_cav_floor
    r0_floor_thk = cm25 + cm0
    r0_floor = layer + r0_floor_con + '\n' + thickness + r0_floor_thk + '\n'
    r0_window = ' WINID=3418 : HINSIDE=11 : HOUTSIDE=64 : SLOPE=-999 : SPACID=4 :\
        WWID=0.77 : WHEIG=1.08 : FFRAME=0.15 : UFRAME=8.17 : ABSFRAME=0.6 : RISHADE=0 :\
        RESHADE=0 : REFLISHADE=0.5 : REFLOSHADE=0.5 : CCISHADE=0.5 : EPSFRAME=0.9 :\
        EPSISHADE=0.9 : ITSHADECLOSE=INPUT 1*SHADE_CLOSE : ITSHADEOPEN=INPUT 1*SHADE_OPEN :\
        FLOWTOAIRNODE=1 : PERT=0 : PENRT=0 : RADMATERIAL=undefined : RADMATERIAL_SHD1=undefined\n'
    
    r1_wall_con = brick + air_cav + fibglass + brick
    r1_wall_thk = cm10 + cm0 + cm6 + cm10
    r1_wall = layer+r1_wall_con +'\n' + thickness+r1_wall_thk + '\n'
    r1_roof_con = plasterboard + air_cav + fibglass + timber + roofdeck + rooftile
    r1_roof_thk = cm3 + cm0 + cm6 + cm5 + cm1 + cm2
    r1_roof = layer + r1_roof_con + '\n' + thickness + r1_roof_thk + '\n'
    r1_floor_con = concrete + air_cav_floor + fibglass
    r1_floor_thk = cm25 + cm0 + cm10
    r1_floor = layer + r1_floor_con + '\n' + thickness + r1_floor_thk + '\n'
    r1_window = ' WINID=3421 : HINSIDE=11 : HOUTSIDE=64 : SLOPE=-999 : SPACID=4 :\
        WWID=0.77 : WHEIG=1.08 : FFRAME=0.15 : UFRAME=8.17 : ABSFRAME=0.6 : RISHADE=0 :\
        RESHADE=0 : REFLISHADE=0.5 : REFLOSHADE=0.5 : CCISHADE=0.5 : EPSFRAME=0.9 :\
        EPSISHADE=0.9 : ITSHADECLOSE=INPUT 1*SHADE_CLOSE : ITSHADEOPEN=INPUT 1*SHADE_OPEN :\
        FLOWTOAIRNODE=1 : PERT=0 : PENRT=0 : RADMATERIAL=undefined : RADMATERIAL_SHD1=undefined\n'
    
    r2_wall_con = brick + air_cav + pu + brick
    r2_wall_thk = cm10 + cm0 + cm10 + cm10
    r2_wall = layer+r2_wall_con +'\n' + thickness+r2_wall_thk + '\n'
    r2_roof_con = plasterboard+ air_cav + pu + timber + roofdeck + rooftile
    r2_roof_thk = cm3 + cm0 + cm15 + cm5 + cm1 + cm2
    r2_roof = layer + r2_roof_con + '\n' + thickness + r2_roof_thk + '\n'
    r2_floor_con = concrete + air_cav_floor + fibglass
    r2_floor_thk = cm25 + cm0 + cm20
    r2_floor = layer + r2_floor_con + '\n' + thickness + r2_floor_thk + '\n'
    r2_window = ' WINID=3202 : HINSIDE=11 : HOUTSIDE=64 : SLOPE=-999 : SPACID=4 :\
        WWID=0.77 : WHEIG=1.08 : FFRAME=0.15 : UFRAME=8.17 : ABSFRAME=0.6 : RISHADE=0 :\
        RESHADE=0 : REFLISHADE=0.5 : REFLOSHADE=0.5 : CCISHADE=0.5 : EPSFRAME=0.9 :\
        EPSISHADE=0.9 : ITSHADECLOSE=INPUT 1*SHADE_CLOSE : ITSHADEOPEN=INPUT 1*SHADE_OPEN :\
        FLOWTOAIRNODE=1 : PERT=0 : PENRT=0 : RADMATERIAL=undefined : RADMATERIAL_SHD1=undefined\n'
    
  
    def change_r(self, house_file, scenario, inf):
        with open(house_file, 'r') as file_in:
            filedata = file_in.readlines()
        old_ext_wall = "CONSTRUCTION EXT_WALL\n"
        old_inf = "INFILTRATION INFIL1\n"
        
        for line in filedata:
            if line == old_inf:
                filedata[filedata.index(line)+1] = " AIRCHANGE=" + str(inf) + '\n'
                print(filedata[filedata.index(line)+1])
            
            elif scenario == 'r0' and line == old_ext_wall:
                filedata[filedata.index(line)+1] = self.r0_wall
                del filedata[filedata.index(line)+2]
                filedata[filedata.index(line)+6] = self.r0_roof
                del filedata[filedata.index(line)+7]
                filedata[filedata.index(line)+11] = self.r0_floor
                del filedata[filedata.index(line)+12]
                filedata[filedata.index(line)+49] = self.r0_window
                print('if1')
            
            elif scenario == 'r1' and line == old_ext_wall:
                filedata[filedata.index(line)+1] = self.r1_wall
                del filedata[filedata.index(line)+2]
                filedata[filedata.index(line)+6] = self.r1_roof
                del filedata[filedata.index(line)+7]
                filedata[filedata.index(line)+11] = self.r1_floor
                del filedata[filedata.index(line)+12]
                filedata[filedata.index(line)+49] = self.r1_window
                print('if2')
            
            elif scenario == 'r2' and line == old_ext_wall:
                filedata[filedata.index(line)+1] = self.r2_wall
                del filedata[filedata.index(line)+2]
                filedata[filedata.index(line)+6] = self.r2_roof
                del filedata[filedata.index(line)+7]
                filedata[filedata.index(line)+11] = self.r2_floor
                del filedata[filedata.index(line)+12]
                filedata[filedata.index(line)+49] = self.r2_window
                print('if3')

            else:
                pass
                    
        with open(house_file, 'w') as housefile_out:
            for line in filedata:
                housefile_out.write(line)