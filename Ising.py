#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:44:07 2016

@author: zhshang
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

L = 10
ESTEP = 1000
STEP = 100000

# Intitialize the Ising Network
def Init():
    return np.ones([L,L])

# Energy Difference, the flip position is at (x,y)
# Make J > 0, which is the ferromagnetic case
def Ediff(Ising,x,y):
    return 2*Ising[x,y]*(Ising[(x-1)%L,y]+Ising[(x+1)%L,y]+Ising[x,(y-1)%L]+Ising[x,(y+1)%L])

def Mdiff(Ising,x,y):
    return -2*Ising[x,y]/(L**2)
    
# Calculate the energy for the Ising system
def EnMag(Ising):
    energy = 0
    mag = 0
    for x in np.arange(L):
        for y in np.arange(L):
            energy = energy - Ising[x,y]*(Ising[(x-1)%L,y]+Ising[(x+1)%L,y]+Ising[x,(y-1)%L]+Ising[x,(y+1)%L])
            mag = mag + Ising[x,y]*1/(L**2)
    return energy*0.5, mag

def Metropolis(T):    
    E_sum = 0
    M_sum = 0
    Esq_sum = 0
    Msq_sum = 0
    Ising = Init()
    # Calculate the physical quantities
    [E,M] = EnMag(Ising)

    for step in np.arange(STEP):
        x = np.random.randint(0,L)
        y = np.random.randint(0,L)
        
        Endiff = Ediff(Ising,x,y)
        Magdiff = Mdiff(Ising,x,y)
        
        if Endiff <= 0:
            Ising[x,y] *= -1
            E += Endiff
            M += Magdiff
        elif np.exp(-Endiff/T) > np.random.rand():
            Ising[x,y] *= -1
            E += Endiff
            M += Magdiff

        if step >= ESTEP:
            E_sum += E
            M_sum += M
            Esq_sum += E**2
            Msq_sum += M**2
        
    E_mean = E_sum/(STEP-ESTEP)/(L**2)
    M_mean = M_sum/(STEP-ESTEP)
    Esq_mean = Esq_sum/(STEP-ESTEP)/(L**4)
    Msq_mean = Msq_sum/(STEP-ESTEP)

    return Ising, E_mean, M_mean, Esq_mean, Msq_mean

M = np.array([])
E = np.array([])
M_sus = np.array([])
SpcH = np.array([])
for T in np.linspace(0.1,5,20):
    [Ising, E_mean, M_mean, Esq_mean, Msq_mean] = Metropolis(T)
    M = np.append(M,np.abs(M_mean))
    E = np.append(E,E_mean)
    M_sus = np.append(M_sus,1/T*(Msq_mean-M_mean**2))
    SpcH = np.append(SpcH,1/T**2*(Esq_mean-E_mean**2))

# plot the figures
T = np.linspace(0.1,5,20)

plt.figure()
plt.plot(T, E, 'rx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle E \rangle$ per site $(J)$')
plt.savefig("E.pdf",format='pdf' ,bbox_inches='tight')

plt.figure()
plt.plot(T, SpcH, 'kx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$C_V$ per site $(\frac{J^2}{k_B^2})$')
plt.savefig("Cv.pdf",format='pdf' ,bbox_inches='tight')

plt.figure()
plt.plot(T, M, 'bx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle|M|\rangle$ per site $(\mu)$')
plt.savefig("M.pdf",format='pdf' ,bbox_inches='tight')

plt.figure()
plt.plot(T, M_sus, 'gx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\chi$ $(\frac{\mu}{k_B})$')
plt.savefig("chi.pdf",format='pdf' ,bbox_inches='tight')

plt.tight_layout()
fig = plt.gcf()
plt.show()

#T = 3.5
#[Ising, E_mean, M_mean, Esq_mean, Msq_mean] = Metropolis(T)
#[E1,M1] = EnMag(Ising)
#E2 = E1/(L**2)
## plot the network cluster
#plt.figure()
#plt.matshow(Ising,cmap='cool')
#plt.axis('off')
