#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhshang
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

L = 16
ESTEP = 1000
STEP = 10000

J = 1 # J>0 to make it ferromagnetic

# Intitialize the XY network
def Init():
    return np.random.rand(L, L)*2*np.pi
    #return np.ones([L, L])

# periodic neighbor
def next(x):
    if x == L-1:
        return 0
    else:
        return x+1

# construct the bond lattice
def FreezeBonds(Ising,T,S):
    iBondFrozen = np.zeros([L,L])
    jBondFrozen = np.zeros([L,L])
    for i in np.arange(L):
        for j in np.arange(L):
            freezProb_nexti = 1 - np.exp(-2 * J * S[i][j] * S[next(i)][j] / T)
            freezProb_nextj = 1 - np.exp(-2 * J * S[i][j] * S[i][next(j)] / T)
            if (Ising[i][j] == Ising[next(i)][j]) and (np.random.rand() < freezProb_nexti):
                iBondFrozen[i][j] = 1
            if (Ising[i][j] == Ising[i][next(j)]) and (np.random.rand() < freezProb_nextj):
                jBondFrozen[i][j] = 1
    return iBondFrozen, jBondFrozen

# H-K algorithm to identify clusters
def properlabel(prp_label,i):
    while prp_label[i] != i:
        i = prp_label[i]
    return i

# Swendsen-Wang cluster
def clusterfind(iBondFrozen,jBondFrozen):
    cluster = np.zeros([L, L])
    prp_label = np.zeros(L**2)
    label = 0
    for i in np.arange(L):
        for j in np.arange(L):
            bonds = 0
            ibonds = np.zeros(4)
            jbonds = np.zeros(4)

            # check to (i-1,j)
            if (i > 0) and iBondFrozen[i-1][j]:
                ibonds[bonds] = i-1
                jbonds[bonds] = j
                bonds += 1
            # (i,j) at i edge, check to (i+1,j)
            if (i == L-1) and iBondFrozen[i][j]:
                ibonds[bonds] = 0
                jbonds[bonds] = j
                bonds += 1
            # check to (i,j-1)
            if (j > 0) and jBondFrozen[i][j-1]:
                ibonds[bonds] = i
                jbonds[bonds] = j-1
                bonds += 1
            # (i,j) at j edge, check to (i,j+1)
            if (j == L-1) and jBondFrozen[i][j]:
                ibonds[bonds] = i
                jbonds[bonds] = 0
                bonds += 1

            # check and label clusters
            if bonds == 0:
                cluster[i][j] = label
                prp_label[label] = label
                label += 1
            else:
                minlabel = label
                for b in np.arange(bonds):
                    plabel = properlabel(prp_label,cluster[ibonds[b]][jbonds[b]])
                    if minlabel > plabel:
                        minlabel = plabel

                cluster[i][j] = minlabel
                # link to the previous labels
                for b in np.arange(bonds):
                    plabel_n = cluster[ibonds[b]][jbonds[b]]
                    prp_label[plabel_n] = minlabel
                    # re-set the labels on connected sites
                    cluster[ibonds[b]][jbonds[b]] = minlabel
    return cluster, prp_label

# flip the cluster spins
def flipCluster(Ising,cluster,prp_label):
    for i in np.arange(L):
        for j in np.arange(L):
            # relabel all the cluster labels with the right ones
            cluster[i][j] = properlabel(prp_label,cluster[i][j])
    sNewChosen = np.zeros(L**2)
    sNew = np.zeros(L**2)
    flips = 0 # get the number of flipped spins to calculate the Endiff and Magdiff
    for i in np.arange(L):
        for j in np.arange(L):
            label = cluster[i][j]
            randn = np.random.rand()
            # mark the flipped label, use this label to flip all the cluster elements with this label
            if (not sNewChosen[label]) and randn < 0.5:
                sNew[label] = +1
                sNewChosen[label] = True
            elif (not sNewChosen[label]) and randn >= 0.5:
                sNew[label] = -1
                sNewChosen[label] = True

            if Ising[i][j] != sNew[label]:
                Ising[i][j] = sNew[label]
                flips += 1

    return Ising,flips

# Swendsen-Wang Algorithm in Ising model (with coupling constant dependency on sites)
# One-step for Ising
def oneMCstepIsing(Ising, S):
    [iBondFrozen, jBondFrozen] = FreezeBonds(Ising, T, S)
    [SWcluster, prp_label] = clusterfind(iBondFrozen, jBondFrozen)
    [Ising, flips] = flipCluster(Ising, SWcluster, prp_label)
    return Ising

# Decompose XY network to two Ising networks with project direction proj
def decompose(XY,proj):
    x = np.cos(XY)
    y = np.sin(XY)
    x_rot = np.multiply(x,np.cos(proj))+np.multiply(y,np.sin(proj))
    y_rot = -np.multiply(x,np.sin(proj))+np.multiply(y,np.cos(proj))
    Isingx = np.sign(x_rot)
    Isingy = np.sign(y_rot)
    S_x = np.absolute(x_rot)
    S_y = np.absolute(y_rot)
    return Isingx, Isingy, S_x, S_y

# Compose two Ising networks to XY network
def compose(Isingx_new,Isingy_new,proj,S_x, S_y):
    x_rot_new = np.multiply(Isingx_new,S_x)
    y_rot_new = np.multiply(Isingy_new,S_y)
    x_new = np.multiply(x_rot_new,np.cos(proj))-np.multiply(y_rot_new,np.sin(proj))
    y_new = np.multiply(x_rot_new,np.sin(proj))+np.multiply(y_rot_new,np.cos(proj))
    XY_new = np.arctan2(y_new,x_new)
    return XY_new

def oneMCstepXY(XY):
    proj = np.random.rand()
    [Isingx, Isingy, S_x, S_y] = decompose(XY, proj)
    Isingx_new = oneMCstepIsing(Isingx, S_x)
    #Isingy_new = oneMCstepIsing(Isingy, S_y)
    # Here we only use the flopping of Ising in projection direction, without the perpendicular direction
    #XY_new = compose(Isingx_new, Isingy_new, proj, S_x, S_y)
    XY_new = compose(Isingx_new, Isingy, proj, S_x, S_y)
    return XY_new

# Calculate the energy for XY network
def EnMag(XY):
    energy = 0
    for i in np.arange(L):
        for j in np.arange(L):
            # energy
            energy = energy - (np.cos(XY[i][j]-XY[(i-1)%L][j])+np.cos(XY[i][j]-XY[(i+1)%L][j])+np.cos(XY[i][j]-XY[i][(j-1)%L])+np.cos(XY[i][j]-XY[i][(j+1)%L]))
    magx = np.sum(np.cos(XY))
    magy = np.sum(np.sin(XY))
    mag = np.array([magx,magy])
    return energy * 0.5, LA.norm(mag)/(L**2)

# Swendsen Wang method for XY model
def SWang(T):
    XY = Init()
    # thermal steps to get the equilibrium
    for step in np.arange(ESTEP):
        XY = oneMCstepXY(XY)
    # finish with thermal equilibrium, and begin to calc observables
    E_sum = 0
    M_sum = 0
    Esq_sum = 0
    Msq_sum = 0
    for step in np.arange(STEP):
        XY = oneMCstepXY(XY)
        [E,M] = EnMag(XY)

        E_sum += E
        M_sum += M
        Esq_sum += E**2
        Msq_sum += M**2

    E_mean = E_sum/STEP/(L**2)
    M_mean = M_sum/STEP
    Esq_mean = Esq_sum/STEP/(L**4)
    Msq_mean = Msq_sum/STEP

    return XY, E_mean, M_mean, Esq_mean, Msq_mean

M = np.array([])
E = np.array([])
M_sus = np.array([])
SpcH = np.array([])
Trange = np.linspace(0.1, 2.5, 10)
for T in Trange:
    [Ising, E_mean, M_mean, Esq_mean, Msq_mean] = SWang(T)
    M = np.append(M, np.abs(M_mean))
    E = np.append(E, E_mean)
    M_sus = np.append(M_sus, 1/T*(Msq_mean-M_mean**2))
    SpcH = np.append(SpcH, 1/T**2*(Esq_mean-E_mean**2))

# plot the figures
T = Trange

plt.figure()
plt.plot(T, E, 'rx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle E \rangle$ per site $(J)$')
plt.savefig("E.pdf", format='pdf', bbox_inches='tight')

plt.figure()
plt.plot(T, SpcH, 'kx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$C_V$ per site $(\frac{J^2}{k_B^2})$')
plt.savefig("Cv.pdf", format='pdf', bbox_inches='tight')

plt.figure()
plt.plot(T, M, 'bx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle|M|\rangle$ per site $(\mu)$')
plt.savefig("M.pdf", format='pdf', bbox_inches='tight')

plt.figure()
plt.plot(T, M_sus, 'gx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\chi$ $(\frac{\mu}{k_B})$')
plt.savefig("chi.pdf", format='pdf', bbox_inches='tight')

plt.tight_layout()
fig = plt.gcf()
plt.show()

#T = 0.1
#[XY, E_mean, M_mean, Esq_mean, Msq_mean] = SWang(T)
#Cv = 1 / T**2 * (Esq_mean - E_mean**2)
#M_sus = 1 / T * (Msq_mean - M_mean**2)
#[E1,M1] = EnMag(XY)
#E2 = E1/(L**2)
#print(E_mean, E2, M_mean, M1, Cv, M_sus)
## plot the network cluster
#plt.figure()
#plt.matshow(XY,cmap='cool')
#plt.axis('off')
