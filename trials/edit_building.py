# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:03:24 2022

@author: 20181270
"""

with open('House_Copy.b18', 'r') as file_in:
    filedata = file_in.readlines()
    
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
cm10 = '0.1\t'
cm15 = '0.15\t'
cm20 = '0.2\t'
cm25 = '0.25\t'

layer = ' LAYERS   = '
thickness = ' THICKNESS= '

base_wall_con =  brick + air_cav + brick
base_wall_thk = cm10 + cm0 + cm10
base_wall = layer + base_wall_con + '\n' + thickness + base_wall_thk + '\n'
base_roof_con = plasterboard+ air_cav + timber + roofdeck + rooftile
base_roof_thk = cm3 + cm0 + cm5 + cm1 + cm2
base_roof = layer + base_roof_con + '\n' + thickness + base_roof_thk + '\n'
base_floor_con = concrete + air_cav_floor
base_floor_thk = cm25 + cm0
base_floor = layer + base_floor_con + '\n' + thickness + base_floor_thk + '\n'
base_window = ' WINID=3418 : HINSIDE=11 : HOUTSIDE=64 : SLOPE=-999 : SPACID=4 :\
    WWID=0.77 : WHEIG=1.08 : FFRAME=0.15 : UFRAME=8.17 : ABSFRAME=0.6 : RISHADE=0 :\
    RESHADE=0 : REFLISHADE=0.5 : REFLOSHADE=0.5 : CCISHADE=0.5 : EPSFRAME=0.9 :\
    EPSISHADE=0.9 : ITSHADECLOSE=INPUT 1*SHADE_CLOSE : ITSHADEOPEN=INPUT 1*SHADE_OPEN :\
    FLOWTOAIRNODE=1 : PERT=0 : PENRT=0 : RADMATERIAL=undefined : RADMATERIAL_SHD1=undefined \n'
    
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

    
def replaceR(cons, filedata):
   pass 
    
#old_ext_wall = "CONSTRUCTION EXT_WALL\nLAYERS\t=WALL_BOARD\tHORIZONTAL\tPOLY_URETH\tCONCRETESL\nTHICKNESS\t=0.01\t0\t0.04\t0.05"
old_ext_wall = "CONSTRUCTION EXT_WALL\n"
for line in filedata:
    if line == old_ext_wall:
        filedata[filedata.index(line)+1] = r1_wall
        del filedata[filedata.index(line)+2]
        filedata[filedata.index(line)+6] = r1_roof
        del filedata[filedata.index(line)+7]
        filedata[filedata.index(line)+11] = r1_floor
        del filedata[filedata.index(line)+12]
        filedata[filedata.index(line)+49] = r1_window
        # del filedata[filedata.index(line)+2]
        
with open('House_copy2.b18', 'w') as housefile_out:
    for line in filedata:
        housefile_out.write(line)