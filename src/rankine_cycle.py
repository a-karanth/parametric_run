# -*- coding: utf-8 -*-
"""
Plot the Rankine cycle on a P-H chart of different refrigerants
Using the package CoolProp

@author: 20181270
"""

from CoolProp.Plots import PropertyPlot
from CoolProp.Plots import SimpleCompressionCycle
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

ref = 'R134a'
# plot = PropertyPlot(ref, 'PH', unit_system='SI')

# Define state points based on evaporator and condenser temperatures
Tcond_inlet = 20
Teva_inlet = -10
Tcond_inlet_K = Tcond_inlet + 273.15  # Condenser inlet temperature, K
Teva_inlet_K = Teva_inlet + 273.15  # Evaporator inlet temperature, K (used for the boiler here)

# State 1: Saturated vapour at evaporator outlet temperature (initial state)
P1 = CP.PropsSI('P', 'T', Teva_inlet_K, 'Q',1 , ref)
h1 = CP.PropsSI('H', 'T', Teva_inlet_K, 'Q', 1, ref)  # Enthalpy at state 1
s1 = CP.PropsSI('S', 'T', Teva_inlet_K, 'Q', 1, ref)

# State 2: Isentropic compression (Pump work)
P2 = CP.PropsSI('P', 'T', Tcond_inlet_K, 'Q', 0, ref)
h2s = CP.PropsSI('H', 'P', P2, 'S', s1, ref)  # Enthalpy at state 2s (isentropic)

# State 3: Saturated vapor at condenser
h3 = CP.PropsSI('H', 'T', Tcond_inlet_K, 'Q', 0, ref)  # Enthalpy at state 3

# State 4: Isenthalpic expansion (through valve, typically, but included here per specification)
h4 = h3  # Enthalpy at state 4, isenthalpic expansion

#%%
plot = PropertyPlot(ref, 'PH', )

# Calculate and draw saturation lines
plot.calc_isolines(CP.iQ, num=11)

# Calculate and draw isotherm lines
plot.calc_isolines(CP.iT, num=25, iso_range=[203, 400])

# Calculate and draw isentropic lines
plot.calc_isolines(CP.iSmass, num=38)

plt.legend()
plt.show()

#plot the custom cycle on the properties chart
ax = plot.axis
H1 = h1/1000; H2 = h2s/1000; H3 = h3/1000; H4 = h4/1000 # converting H from j/kg to kJ/kg
p1 = P1/1000; p2 = P2/1000 # converting pressure from pascal to kilopascal
ax.plot([H1,H2,H3,H4,H1],[p1,p2,p2,p1,p1], marker='o', label='custom cycle')
ax.legend()
# Show the plot
plot.show()
