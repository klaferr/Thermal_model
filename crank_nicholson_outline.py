#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:00:37 2021

@author: laferrierek

This is the origin file for the thermal 1D semi-implict crank-nicoslon code that I will eventually write

Builds off notes+papers provided by A. Bramson

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import constants as const
from orbitial_parameters import orbital_params as op     # fix this 
import json
from scipy import sparse
from scipy import linalg as linalg
from alive_progress import alive_bar

    
loc = '/Users/laferrierek/Desktop/Mars_Troughs/Codes/thermal_model/'
#%% Aesthetic
fs = (10, 8)
res = 350
plt.rc("font", size=18, family="serif")
plt.style.use('ggplot')

#%% Constants - pull the specific mars ones from a paramfille. 
# Constants
G = 6.67259*10**(-11)       # Gravitational Constant; units
kb = 1.38*10**(-23)         # ; J/K
NA = 6.022 * 10**(23)       # per mol
h = const.h
c = const.c
k = const.k
b = 2.89777199*10**(-3)     # meters
sigma = 5.670*10**(-8)      # Stefan-Boltzmann; W/m^2K^4
au = 1.4959787061e11        # AU; meters
sm  = 1.9891e30             # Solar Mass; kg

# conversion
mbar_to_Pascal = 100
gpercm3_tokgperm3 = 1000

# Molecules
hydrogen = 1.004                    # g/mol
oxygen = 15.999                     # g/mol
carbon = 12.01                      # g/mol
# H2O
m_gas_h2o = (hydrogen*2+oxygen)/1000    # kg/mol
triple_P_h2o = 611.657                  # Pa
triple_Temp_h2o = 273.1575              # K
Lc_H2O = 2257*10**3                     # J/kg

# CO2
m_gas_co2 = (carbon+oxygen*2)/1000  # kg/mol
triple_P_co2 = 516757               # Pa
triple_Temp_co2 = 216.55            # K
Lc_CO2 =  589.9*10**3                 # Latent heat of CO2 frost; J/kg
CO2_FrostPoints = 150

# Earth Specific constants
EarthYearLength = 2*np.pi*np.sqrt(au**3/(G*sm))             # Length of one Earth year in seconds
solarflux_at_1AU = 1367                                     # Current; W/m2

# Mars Specific constants
Mars_semimajor = 1.52366231                                # Distance; AU
MarsyearLength = 2*np.pi/np.sqrt(G*sm/(Mars_semimajor*au)**3)   # Length of Mars year in seconds using Kepler's 3rd law
MarsdayLength = MarsyearLength/365 # this is going to be more complex since the insolation changes.   
solarflux_at_Mars = solarflux_at_1AU/Mars_semimajor**2

# Thermal (Bramosn et al. 2017, JGR Planets)
# compositional values - this may need to be read in

albedo = 0.25
emissivity = 1.0
Q = 30*10**(-3)
Tref = 400 #230

# Run stuff
runTime = 15
f = 0.5
dt = 500

thermal_conductivity_rock = 2           # W/mK
thermal_conductivity_ice = 3.2
thermal_conductivity_air = 0

density_rock = 3300                 # kg/m3
density_ice = 920
density_air = 0

cp_rock = 837                   # J/kgK
cp_ice = 1540
cp_air = 0


# Frost
emisFrost = 0.95
albedoFrost = 0.6
Tfrost = CO2_FrostPoints
windupTime = 8
convergeT = 0.01

#%% Read in values for trough, get orbitial solutions
class Profile:
  def reader(self,input_dict,*kwargs):
    for key in input_dict:
      try:
        setattr(self, key, input_dict[key])
      except:
        print("no such attribute, please consider add it at init")
        continue
    
with open(loc+"trough_parameters.json",'r') as file:
  a=file.readlines()

Mars_Trough=Profile()
Mars_Trough.reader(json.loads(a[0]))

# Orbital solutions
eccentricity = 0.09341233
obl = np.deg2rad(25.19)
Lsp = np.deg2rad(250.87)
dt_orb = 500
    
soldist, sf, IRdown, visScattered, nStepsInYear, lsWrapped, hr, ltst, lsrad, az, sky, flatVis, flatIR = op(eccentricity, obl, Lsp, dt_orb, Mars_Trough)
# none were nan values
# plot checks
plt.plot(lsWrapped, soldist, '.')
plt.ylabel('sol dist')
plt.xlabel('lswrapped')
plt.show()

plt.plot(sf)
plt.ylabel('sf')
plt.xlabel('total steps')
plt.show()

plt.plot(lsWrapped, sf)
plt.ylabel('sf')
plt.xlabel('ls Wrapped')
plt.show()

plt.plot(hr, sf, '.')
plt.ylabel('sf')
plt.xlabel('hr angle')
plt.show()

plt.plot(lsWrapped, IRdown, '.', label='IR')
plt.plot(lsWrapped, visScattered, '.', label='Vis')
plt.legend()
plt.ylabel('Irdown and visscat')
plt.xlabel('LsWrapped')
plt.show()


#%% Functions 
# - Phase diagrams
def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))

def mean_gas_const(f1, m1, f2, m2, f3, m3):
    '''
    f's are the % volume of the atmosphere as a fraction
    m's are the molar mass of the molecule (sometimes a compound)
    '''
    mbar = f1*m1+f2*m2+f3*m3
    return 8.314/mbar

# Thermal - Bramson et al. 2017
def thermal_diffusivity_calc(k, rho, cp):
    return k/(rho*cp)

def thermal_skin_depth(k, rho, cp, P):
    thermal_diff = thermal_diffusivity_calc(k, rho, cp)
    skin = np.sqrt(4*np.pi*thermal_diff*P)
    return skin
    
def surface_enegry_balance(Solar, incidence_angle, albedo, dmco2_dt, dTemp_dz, IR_downwelling):
    T_surface = ((Solar*np.cos(incidence_angle)*(1-albedo)+Lc_CO2 * dmco2_dt +k*dTemp_dz*IR_downwelling)/(emissivity*sigma))**(1/4)
    return T_surface

def stability_depth(triple_pressure, triple_T, Lc, T, f1, m1, rho_atmo, layer_depth):
    R_bar = mean_gas_const(f1, m1, 0, 0, 0, 0)
    pressure_sublimation = clapyeron(triple_pressure, triple_T, R_bar, Lc, T)
    rho_vapor = pressure_sublimation/(R_bar*T)
    match = np.argwhere(rho_vapor >= rho_atmo)[0]
    excess_ice_depth = layer_depth[match[0]]
    return excess_ice_depth

# - Plots
def plot_layers(layer_depth, layer_number, layer_thickness):
    plt.rc("font", size=18, family="serif")
    plt.figure(figsize=(10,10), dpi=res)
    subsurface = np.linspace(0, np.nanmax(layer_depth)+5, (np.int(np.nanmax(layer_depth)+5)))
    image = np.ones((np.int(np.nanmax(layer_depth)+5), 20))*np.reshape(subsurface, ((np.int(np.nanmax(layer_depth)+5)), 1))
    layer_number = np.arange(1, 16, 1)
    plt.imshow(image, vmin=0, vmax=(np.int(np.nanmax(layer_depth)+5)), cmap='summer')
    plt.colorbar(label='Depth (cm)')
    plt.scatter(np.ones((15))*10, layer_depth, c='k', marker='o')
    for i in range(0, 15):
        plt.annotate('%2.0f'%(layer_number[i]), (10.5, layer_depth[i]+0.5))
    plt.hlines(layer_thickness/2+layer_depth, 0, 20, colors='k', linestyle='dashed')
    plt.hlines(layer_depth[0]-layer_thickness[0]/2, 0, 20, colors='r', linestyle='dashed')
    plt.ylim((-0.1, (np.int(np.nanmax(layer_depth)+3))))
    plt.xlim((0, 19))
    plt.gca().invert_yaxis()
    plt.show()

#%% Layers - define from matlab code, changes at depth


def layerProperties(layerPropertyVectors, lengthOfYear, lengthOfDay, layerGrowth, dailyLayers, annualLayers):
    # Set up vectors for each layer's properties

    
    k_input = layerPropertyVectors[:, 0]
    density_input = layerPropertyVectors[:, 1]
    c_input = layerPropertyVectors[:, 2]
    depths_input = layerPropertyVectors[:, 3]
    
    nPropLayers = np.size(k_input)
    
    allKappas = k_input / (density_input * c_input)
    
    diurnalSkinDepths = np.sqrt(allKappas*lengthOfDay/np.pi) #% meters
    print('Diurnal thermal skin depth of top layer = %.8f m'%diurnalSkinDepths[0])
    annualSkinDepths = np.sqrt(allKappas*lengthOfYear/np.pi) #% meters
    print ('Annual thermal skin depth of bottom layer = %.8f m'%annualSkinDepths[-1])
      
    firstLayerThickness = diurnalSkinDepths[0] / ((1-layerGrowth**dailyLayers)/(1-layerGrowth)) #% Thickness of first layer based on diurnal skin depth of surface material
    numberLayers = math.ceil(np.log(1-(1-layerGrowth)*(annualLayers*annualSkinDepths[-1]/firstLayerThickness))/np.log(layerGrowth) ) #% Number of subsurface layers based on annual skin depth of deepest layer
    
    dz = (firstLayerThickness * layerGrowth**np.arange(0, numberLayers, 1)) #% transpose to make column vector
    depthsAtMiddleOfLayers = np.cumsum(dz) - dz/2
    depthsAtLayerBoundaries = np.cumsum(dz)
    depthBottom = sum(dz)
 
    k_vector = np.zeros(numberLayers)       #% Thermal conductivities (W m^-1 K^-1)
    density_vector = np.zeros(numberLayers) #% Densities of subsurface (kg/m^3)
    c_vector = np.zeros(numberLayers)       #% Specific heats J/(kg K)
    
    for ii in range(0, nPropLayers):
        if depths_input[ii] > depthBottom:
            print('Warning: Model domain isn''t deep enough to have a layer at %f m.' %(depths_input[ii]))
        else:
            nPropLayersToUse = ii

    
    layerIndices = np.zeros(nPropLayersToUse+1) #% numerical layer index for top of each layer
    for ii in range(0, nPropLayersToUse+1):
                
        if ii==0:
            
            indexStart = 0
            
            if ii+1 <= nPropLayersToUse:
                indexesBelowDepth1 = np.argwhere(depthsAtMiddleOfLayers<depths_input[ii+1])
                indexEnd = indexesBelowDepth1[-1][0]
            else:
                indexEnd = numberLayers[0]
            
        else:
            indexesBelowDepth2 = np.argwhere(depthsAtMiddleOfLayers<depths_input[ii])
            indexStarta = indexesBelowDepth2[-1] 
            indexStart = indexStarta[0]
            if ii+1 <= nPropLayersToUse:
                indexesBelowDepth3 = np.argwhere(depthsAtMiddleOfLayers<depths_input[ii+1])
                indexEnd = indexesBelowDepth3[-1][0]

            else:
                indexEnd = numberLayers

                        
        k_vector[indexStart:indexEnd] = k_input[ii]
        density_vector[indexStart:indexEnd] = density_input[ii]
        c_vector[indexStart:indexEnd] = c_input[ii]
        layerIndices[ii] = indexStart
     
    #% Calculate diffusivity
    Kappa_vector = k_vector / (density_vector * c_vector)
    
    #% Calculate timestep needed to fulfill Courant Criterion for
    #% numerical stability
    courantCriteria = dz * dz / (5 * Kappa_vector)
    courantdt = min(courantCriteria)
    print('For numerical stability, delta t will be %2.5f s.'%courantdt)

    modelLayers = np.array([k_vector, density_vector, c_vector, Kappa_vector, dz, depthsAtMiddleOfLayers])
    dt = courantdt

    return  modelLayers, dt, layerIndices

k_input = np.array([0.0459, 2.952])  
density_input = np.array([1626.18, 1615])  
c_input = np.array([837, 925]) 
depths_input = np.array([0, 0.5])
layerGrowth = 1.03                
dailyLayers  = 10                  
annualLayers = 6   

layerPropertyVectors = np.vstack((k_input, density_input, c_input, depths_input)).T

modelLayers, dt, layerIndicies = layerProperties(layerPropertyVectors, MarsyearLength, Mars_Trough.Rotation_rate, layerGrowth,dailyLayers, annualLayers)
#%%
"""
 # layers have to cover the top 30 meters.
# Define layers - move to outside , need to do next 15 layers
# 15 within 1 skin depth (defines for 1 year), with increasing size by 3%
nLayers = 15
Length = MarsyearLength*10 
annualskindepth = thermal_skin_depth(thermal_conductivity_ice, density_ice, cp_ice, Length)
layers = np.array([1.03**i for i in range(0, nLayers)])
layer_thickness = annualskindepth*layers/(np.sum(layers))
layer_number = np.arange(0, nLayers, 1)
layer_depth = np.array([np.sum(layer_thickness[0:i])+layer_thickness[i]/2 for i in range(np.size(layer_thickness))])
nLayers = np.size(layer_number)
dz = layer_thickness

depthsAtMiddleOfLayers = layer_depth

print('Annual skin depth: %2.2f m' %annualskindepth)

#%%
# plot check the layer distribution
# draw out - i've done this before.
background = np.reshape(np.arange(0, 42, 1), (42, 1))
background=background*np.ones((42, 10))
plt.figure(figsize=(5, 5), dpi=120)
plt.imshow(background)
plt.scatter(np.ones((15))*5, layer_depth, c='red')
#plt.imshow(np.reshape(np.arange((0, 42, 1), ((42, 1)))*np.ones((42, 10))))
#plt.scatter(np.ones((15)), layer_depth,c='r')
plt.hlines((layer_thickness/2 +layer_depth), 0, 10,  color='k', linestyle='dashed')
plt.hlines((layer_depth[0] - layer_thickness[0]/2 ), 0, 10,  color='k', linestyle='dashed')
for i in range(nLayers):
    plt.annotate('%2.0f'%(i+1), (5.5, layer_depth[i]+0.3), c='k' )
plt.ylim((0, 41))
plt.xlim((0, 9))
plt.gca().invert_yaxis()
plt.colorbar(label='Composition (not actual)')
plt.show()
#%%
kthermal = thermal_conductivity_rock*np.ones((nLayers))
rho = density_rock*np.ones((nLayers))
cp = cp_rock*np.ones(nLayers)

kappa  = kthermal/(rho*cp)
timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
"""
#%% read in layers from file
# if we are to make layers such that they can be changed within the code, (which will be added at a later date), 
# then  it is necessary to have ktherm cp, etc be changable. 
#ktherm, cp, rho, dz = np.loadtxt(loc+'/Layer_properties_Mars.txt', unpack=True, delimiter=',')
                                 
#%% potential fast solver?
import numpy as np

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

#%% Define as fuhnction with inputs

def Crank_Nicholson(nLayers, nStepsInYear, windupTime, runTime, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref):
    Temps = np.zeros((nLayers, nStepsInYear))
    Tsurf = np.zeros((nStepsInYear))
    lastTimestepTemps = np.zeros((nLayers, runTime))
    oldTemps = np.zeros((nLayers))
    oldTemps[:] = Tref
    print('Initial Surface Temperature = %2.2f K' %Tref)
    #ktherm_st = ktherm
    
    # Define alpha values for matrix
    #alpha_u = (2*ktherm_st*np.roll(ktherm_st,1)/(ktherm_st*np.roll(dz,1) + np.roll(ktherm_st, 1)*dz))*(dt/(rho*cp*dz))
    alpha_u = (2*ktherm*np.roll(ktherm,1)/(ktherm*np.roll(dz,1) + np.roll(ktherm, 1)*dz))*(dt/(rho*cp*dz))
    alpha_u[0] = 0
    alpha_d = (2*ktherm*np.roll(ktherm,-1)/(ktherm*np.roll(dz,-1) +np.roll(ktherm,-1)*dz))*(dt/(rho*cp*dz))
    alpha_d[-1] = 0
    
    #define diagnols, e for explicit, i for implicit
    dia_e = np.zeros((nLayers, 1))
    dia_e = 1 - (1-f)*alpha_u - (1-f)*alpha_d
    dia_e[nLayers-1] = 1 - (1-f)*alpha_u[-1]
    
    dia_i = np.zeros((nLayers, 1));
    dia_i = 1 + f*alpha_u + f*alpha_d;
    dia_i[nLayers-1] = 1+f*alpha_u[-1];
    
    # Boundary conditions
    boundary = np.zeros((nLayers))
    boundary[-1] = dt*Q/(rho[-1]*cp[-1]*dz[-1])
    
    B_implicit = np.array([(-f*alpha_u), (1+f*alpha_u + f*alpha_d), (-f*alpha_d)])
    Amatrix_i = sparse.spdiags(B_implicit, [-1, 0, 1], nLayers, nLayers)
    A = Amatrix_i.toarray()
    #anew = -f*alpha_u
    #bnew = 1+f*alpha_u + f*alpha_d
    #cnew = -f*alpha_d
    
    #beta = ktherm_st[0]*dt/(rho[0]*cp[0]*dz[0]*dz[0])
    beta = ktherm[0]*dt/(rho[0]*cp[0]*dz[0]*dz[0])

    # I excluded sf_i, IRdown,, viscattered, and T_e
            
    # Total fluxes
    Fin = (sf + visScattered*sky + flatVis*(1-sky))*(1-albedo) + (IRdown*sky + flatIR*(1-sky))*emissivity;
    Fin_i = (np.roll(sf, -1) + np.roll(visScattered, -1)*sky + np.roll(flatVis, -1)*(1-sky))*(1-albedo) + (np.roll(IRdown, -1)*sky + np.roll(flatIR, -1)*(1-sky))*emissivity
    Fin_frost = (sf + visScattered*sky +flatVis*(1-sky))*(1-albedoFrost)+(IRdown*sky + flatIR*(1-sky))*emisFrost
    # Calculate a and b's for surface temperature calculation
    #aa = (dz[0]/(2*ktherm_st[0])*(Fin[0] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm_st[0]))))
    #b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm_st[0])))
    
    aa = (dz[0]/(2*ktherm[0])*(Fin[0] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
    b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0])))
    Tsurf[0] = aa+b*Tref
    
    # Frost mass
    #gamma_frost = (-1/Lc_CO2)*(2*ktherm_st[0]*(dt/dz[0]))
    #theta_frost = (dt/Lc_CO2)*(2*ktherm_st[0]*CO2_FrostPoints/dz[0] - Fin_frost +emisFrost*sigma*CO2_FrostPoints**4)
    gamma_frost = (-1/Lc_CO2)*(2*ktherm[0]*(dt/dz[0]))
    theta_frost = (dt/Lc_CO2)*(2*ktherm[0]*CO2_FrostPoints/dz[0] - Fin_frost +emisFrost*sigma*CO2_FrostPoints**4)
    
    theta_frost_i = np.roll(theta_frost, -1)
    defrosting_decrease = np.exp(-depthsAtMiddleOfLayers/timestepSkinDepth)
    
    frostMass = 0
    frostMasses = np.zeros((nStepsInYear))
    
    #a_ee = (dz[0]/(2*ktherm[0]))*(Fin[n] + 3*emissivity*sigma*Tref**4/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
    
    #with alive_bar(runTime) as bar:  # declare your expected total
    for yr in range(0, runTime):  # this is the run up time before actually starting.   
        for n in range(0, nStepsInYear):
            if frostMass == 0:
                # Have to recacluate each time   
                #ktherm_t = ktherm #ktherm(ka, kr, oldTemps)
                #b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm_t[0])))
                #a_e = (dz[0]/(2*ktherm_t[0]))*(Fin[n] + 3*emissivity*sigma*Tref**4/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm_t[0]))))
                #a_i = (dz[0]/(2*ktherm_t[0])*(Fin_i[n] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm_t[0]))))
                
                b = 1/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0])))
                a_e = (dz[0]/(2*ktherm[0]))*(Fin[n] + 3*emissivity*sigma*Tref**4/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                a_i = (dz[0]/(2*ktherm[0])*(Fin_i[n] + 3*emissivity*sigma*Tref**4)/(1+(4*emissivity*sigma*Tref**3*dz[0]/(2*ktherm[0]))))
                
                boundary[0] = 2*beta*((1-f)*a_e + f*a_i)
          
                # Explicit Part
                dia_e[0] = 1 - (1-f)*(alpha_d[0]+(2-2*b)*beta)
                T_e = (1-f)*alpha_u*np.roll(oldTemps,1) + (1-f)*alpha_d*np.roll(oldTemps,-1) + dia_e*oldTemps + boundary
        
                # Implicit Part - can't this be doen outside?
                A[0, 0] = 1 + f*(alpha_d[0]+(2-2*b)*beta)
                B = T_e
                
                # find new temps
                #anew = -f*alpha_u
                #bnew = 1+f*alpha_u + f*alpha_d
                #cnew = -f*alpha_d
                #bnew[0] = 1 + f*(alpha_d[0]+(2-2*b)*beta)
                #dnew = T_e
                #Temps[:, n] = TDMAsolver(anew,bnew,cnew,dnew)
                Temps[:, n] = np.linalg.solve(A, B)
                #print('Linalg, %2.9f'%(time.time()-st))
                Tsurf[n] = a_i + b*Temps[0,n]  # Uses implicit a with new T calculation- instantanous balance
                frostMass = 0
                if Tsurf[n] < Tfrost:
                    deltaTsurf = Tfrost - Tsurf[n]
                    frostMass = deltaTsurf*rho[0]*cp[0]*timestepSkinDepth/Lc_CO2
                    Temps[:, n] = Temps[:, n] +deltaTsurf*defrosting_decrease
                    Tsurf[n] = Tfrost
            elif frostMass > 0:
                #print('Frost mass >0', n)
                boundary[0] = 2*beta*Tfrost
                
                # Explicit Part
                #b = 0
                dia_e[0] = 1 - (1-f)*(alpha_d[0]+(2)*beta)
                T_e = (1-f)*alpha_u*np.roll(oldTemps,1) + (1-f)*alpha_d*np.roll(oldTemps,-1) + dia_e*oldTemps + boundary
        
                # Implicit Part - can't this be doen outside?
                A[0, 0] = 1 + f*(alpha_d[0]+2*beta)
                B = T_e

                #bnew[0] = 1 + f*(alpha_d[0]+(2-2*b)*beta)
                #dnew = T_e
                #Temps[:, n] = TDMAsolver(anew,bnew,cnew,dnew)
                
                Temps[:, n] = np.linalg.solve(A,B)
                Tsurf[n] = Tfrost
                
                frostMass = frostMass + (1-f)*gamma_frost*oldTemps[0] +theta_frost[n] + f*(gamma_frost*Temps[0, n] + theta_frost_i[n]) 
                
                if frostMass < 0:
                    shiftedFrostMasses = np.roll(frostMasses, 1)
                    timeDefrosted = np.sqrt((0-frostMass)/shiftedFrostMasses[n] -frostMass)
                    deltaTsurf2 = -frostMass*Lc_CO2/(rho[0]*cp[0]*timestepSkinDepth*timeDefrosted)
                    Tsurf[n] = Tfrost+deltaTsurf2
                    Temps[:, n] = Temps[:, n]+deltaTsurf2*defrosting_decrease
                    frostMass = 0
            else:
                print('Frost mass is negative, issue', n)
            oldTemps[:] = Temps[:, n]
            Tref = Tsurf[n]
            frostMasses[n] = frostMass
        lastTimestepTemps[:,yr] = Temps[:,n]  # To compare for convergence 
        print('Youre %2.0f / %2.0f'%(yr, runTime))
        if yr == windupTime:
            windupTemps = np.nanmean(Tsurf)
            oldTemps[:] = windupTemps
            print('Windup done, Setting all temps to %4.2f'%windupTemps)
        if yr == runTime-1:
            tempDiffs = lastTimestepTemps[:, runTime-1] -lastTimestepTemps[:, runTime-2]
            whichConverge = np.abs(tempDiffs) < convergeT
            if np.sum(whichConverge) == np.size(whichConverge):
                print('Converge to %3.7f'%(np.max(np.abs(tempDiffs))))
            else:
                print('Did not converge, increase run Time')
        if yr > 1:
            tempDiffs = lastTimestepTemps[:,yr] - lastTimestepTemps[:,yr-1]
            print('Still at least %3.7f K off' %np.max(np.abs(tempDiffs)))
            #print(lastTimestepTemps[:, yr], lastTimestepTemps[:, yr-1])
        # If you want progress bar to work, must run in terminal. 
        #bar()                 
    return Temps, windupTemps #, Tsurf, frostMasses


#%% Layers
 # layers have to cover the top 30 meters.
# Define layers - move to outside , need to do next 15 layers
# 15 within 1 skin depth (defines for 1 year), with increasing size by 3%
nLayers = 15
Length = MarsyearLength 
annualskindepth = thermal_skin_depth(thermal_conductivity_ice, 1615, cp_ice, Length)
layers = np.array([1.03**i for i in range(0, nLayers)])
layer_thickness = annualskindepth*layers/(np.sum(layers))
layer_number = np.arange(0, nLayers, 1)
layer_depth = np.array([np.sum(layer_thickness[0:i])+layer_thickness[i]/2 for i in range(np.size(layer_thickness))])
nLayers = np.size(layer_number)

depthsAtMiddleOfLayers = layer_depth

print('Annual skin depth: %2.2f m' %annualskindepth)

# plot check the layer distribution
# draw out - i've done this before.
background = np.reshape(np.arange(0, 42, 1), (42, 1))
background=background*np.ones((42, 10))
plt.figure(figsize=(5, 5), dpi=120)
plt.imshow(background)
plt.scatter(np.ones((15))*5, layer_depth, c='red')
#plt.imshow(np.reshape(np.arange((0, 42, 1), ((42, 1)))*np.ones((42, 10))))
#plt.scatter(np.ones((15)), layer_depth,c='r')
plt.hlines((layer_thickness/2 +layer_depth), 0, 10,  color='k', linestyle='dashed')
plt.hlines((layer_depth[0] - layer_thickness[0]/2 ), 0, 10,  color='k', linestyle='dashed')
for i in range(nLayers):
    plt.annotate('%2.0f'%(i+1), (5.5, layer_depth[i]+0.3), c='k' )
plt.ylim((0, 41))
plt.xlim((0, 9))
plt.gca().invert_yaxis()
plt.colorbar(label='Composition (not actual)')
plt.show()
#%%
ktherm = thermal_conductivity_rock*np.ones((nLayers))
rho = density_rock*np.ones((nLayers))
cp = cp_rock*np.ones(nLayers)
kappa  = ktherm/(rho*cp)
dz = layer_thickness
depthsAtMiddleOfLayers = layer_depth
timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
nLayers = 15

#%% run - issue with dz change not effecting
# add variable ktherm within function
#kr = 0
#dt = 15
ktherm = modelLayers[0, :]
rho = modelLayers[1, :]
cp = modelLayers[2, :]
kappa = modelLayers[3, :]
dz = modelLayers[4, :]
depthsAtMiddleOfLayers = modelLayers[5, :]
timestepSkinDepth = np.sqrt(kappa[0]*dt/np.pi)
#dt = 1000
nLayers = np.size(dz)
#%%
import time
st = time.time()
Temps15, windupTemps15 = Crank_Nicholson(nLayers, nStepsInYear, windupTime, 15, ktherm, dz, dt, rho, cp, emissivity, Tfrost, 120)
#Temps150, windupTemps150 = Crank_Nicholson(nLayers, nStepsInYear, windupTime, 150, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)
print('Took %2.2f'%(time.time()-st))
# check that matches from test and funcrtion. 
#%% Plot checks - i think temperature issues are rleated to the layer formation - not the thermal conductivity. 

# temps is output as a [15, time]
# plotting each layer at specific time requires calling both parts. 

# Depth versus Temperature with intervals of t_day
#t_day = 50 # days
plt.figure(figsize=(5, 5), dpi=300)
#Depths = np.arange(0, np.size(modelLayers[0, :]), 1)
for i in range(0, np.size(Temps15[0, :])):
    if i ==1: # this has no meaning other than it should be about 6 lines
        plt.plot#(Temps15[:, i], modelLayers[-1, :], '.', c='k')
        plt.plot(Temps15[:, i], layer_depth, '.', c='k')

    elif i % 19796 == 0:
        #plt.plot(Temps15[:, i], modelLayers[-1, :],  c='b')
        plt.plot(Temps15[:, i], layer_depth,  c='b')

plt.gca().invert_yaxis()
plt.ylabel('Depth (m)')
plt.xlabel('Temperature (K)')
plt.show()

# make logical sense; included in the outpiut of crank nicholson is teh tempeature value per layer at each time
# plotitng hthe depth requires asuming from the model layers
# plotitng hte tempeatures comes from that output, but assign to depth value., 

#%%
# add variable ktherm within function
import time
st = time.time()
Temps15, windupTemps15 = Crank_Nicholson(nLayers, nStepsInYear, windupTime, 15, ktherm, dz, dt, rho, cp, emissivity, Tfrost, 250) #Tref)
print('Took %2.2f'%(time.time()-st))
#%%
#Calculate and print some output
Temps, windupTemps, Tsurf, frostMasses = Crank_Nicholson(nLayers, nStepsInYear, windupTime, 15, ktherm, dz, dt, rho, cp, emissivity, Tfrost, Tref)

# Find min, max and average temperatures at each depth over the last year
print('Minimum Surface Temp: %8.4f K\n', np.min(Tsurf))
print('Maximum Surface Temp: %8.4f K\n', np.max(Tsurf))
print('Mean Surface Temp: %8.4f K\n', np.nanmean(Tsurf))

minT = np.min(Temps)
maxT =  np.max(Temps)
averageTemps = np.mean(Temps)

rho_CO2ice = 1600
equivalentCO2Thicknesses = frostMasses/rho_CO2ice;

print('Max frost thickness during the year: %5.4f m.\n', max(equivalentCO2Thicknesses))

print('Minimum Frost Mass: %8.4f kg/m^2\n', min(frostMasses))
print('Maximum Frost Mass: %8.4f kg/m^2\n', max(frostMasses))
print('Mean Frost Mass: %8.4f kg/m^2\n', np.nanmean(frostMasses))


#%% Create diurnal averages throughout the last year
beginDayIndex = []
beginDayIndex[0] = 1
dayIndex = 2
for n in range(2, np.size(hr)): # 2:size(hr, 1):
    if h[n] > 0 & hr[n-1] < 0:
        beginDayIndex[dayIndex] = n
        dayIndex = dayIndex + 1

numDays = max(np.size(beginDayIndex))
averageDiurnalTemps = np.zeros(nLayers, numDays)
averageDiurnalSurfTemps = np.zeros(numDays)

for n in numDays:
    if n == numDays:
        averageDiurnalTemps[:,n] = np.nanmean(Temps[:, beginDayIndex[n]:np.size(Temps)])
        averageDiurnalSurfTemps[n] = np.nanmean(Temps[beginDayIndex[n]:np.size(Temps)])
    else:
        averageDiurnalTemps[:,n] = np.nanmean(Temps[:, beginDayIndex[n]:beginDayIndex[n+1]-1])
        averageDiurnalSurfTemps[n] = np.nanmean(Temps[beginDayIndex[n]:beginDayIndex[n+1]-1])

averageDiurnalAllTemps = np.concatenate(averageDiurnalSurfTemps, averageDiurnalTemps)

#%% Calculate ice table/top of subsurface layer temperatures
numPropLayers = np.size(ktherm)
iceTableIndex = 2
if numPropLayers > 1:
    
    iceTableTemps = (ktherm[iceTableIndex]*dz[iceTableIndex-1]*Temps[iceTableIndex,:] + ktherm[iceTableIndex-1]*dz[iceTableIndex]*Temps[iceTableIndex-1,:])/(ktherm[iceTableIndex]*dz[iceTableIndex-1] + k[iceTableIndex-1]*dz[iceTableIndex])
    iceTableTemps = iceTableTemps
    
    iceTable_Pv = 611 * np.exp( (-51058/8.31)*(1/iceTableTemps - 1/273.16) ) # compute vapor pressure at ice table
    iceTable_rhov = iceTable_Pv * (0.01801528 / 6.022140857e23) / (1.38064852e-23 * iceTableTemps); # compute vapor densities at ice table
    
    meanIceTable_Pv = np.nanmean(iceTable_Pv) # get mean for that year of ice table vapor pressures
    meanIceTable_rhov = np.nanmean(iceTable_rhov) # get mean for that year of ice table vapor densities
    meanIceTableT = np.nanmean(iceTableTemps) # mean temperature at the ice table over the year
    meansurfT = np.nanmean(Tsurf) # mean surface temperature

