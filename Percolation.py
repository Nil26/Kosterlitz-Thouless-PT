#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:38:57 2016

@author: zhshang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def ConstructNetwork(N,p):
    return np.less_equal(np.random.rand(N**2),p)

def find(x,prp_label):
    y = x
    while (prp_label[y] != y):
        y = prp_label[y]
    while (prp_label[x] != x):
        z = prp_label[x]
        prp_label[x] = y
        x = z
    return y

def union(x,y,prp_label):
    prp_label[find(x,prp_label)] = find(y,prp_label)
    
def ClusterLabel(N,Network):
    Label = np.zeros(N**2)
    label_index = 0
    prp_label = np.array([-1])
    for i in np.arange(N**2):
        if np.bool(Network[i]): # if the site is occupied
            left = i-1
            above = i-N
            # deal with edge points
            if (i==0): # Starting site
                label_index = label_index + 1
                Label[i] = label_index
                prp_label = np.append(prp_label, np.array([label_index]))
            elif (i <=  N-1): # Top edge
                if np.bool(Network[left]):
                    Label[i] = find(Label[left],prp_label)
                else:
                    label_index = label_index + 1
                    Label[i] = label_index
                    prp_label = np.append(prp_label, np.array([label_index]))
            elif (i%N == 0): # Left edge
                if np.bool(Network[above]):
                    Label[i] = find(Label[above],prp_label)
                else:
                    label_index = label_index + 1
                    Label[i] = label_index
                    prp_label = np.append(prp_label, np.array([label_index]))
            elif (not np.bool(Network[left])) and (not np.bool(Network[above])): # non of the neighbors labeled
                label_index = label_index+1
                Label[i] = label_index
                prp_label = np.append(prp_label,np.array([label_index]))
            elif np.bool(Network[left]) and (not np.bool(Network[above])): # left only occupied
                Label[i] = find(Label[left],prp_label)
            elif (not np.bool(Network[left])) and np.bool(Network[above]): # above only occupied
                Label[i] = find(Label[above],prp_label)
            elif np.bool(Network[left]) and np.bool(Network[above]): # both left and above occupied
                if Label[left] == Label[above]:
                    Label[i] = Label[left]
                else:
                    index_smaller = np.min([Label[left],Label[above]])
                    index_larger = np.max([Label[left],Label[above]])
                    union(index_larger,index_smaller,prp_label)
                    Label[i] = find(index_smaller,prp_label)
    return [Label,prp_label]
    
def relabel(Label,prp_label,N):
    for i in np.arange(N**2):
        if (np.bool(Network[i])):
            y = Label[i]
            while prp_label[y] != y:
                y = prp_label[y]
            Label[i] = y
    return Label    

def spanning(label_re,N):
    top = label[0:N]
    bottom = label[-N:]
    IF = False
    for i in top:
        if i != 0:
            IF = IF or (i in bottom)
    return IF            

# Problem 1.2
N = 64
Network = ConstructNetwork(N,0.58)
[label,prp_label] = ClusterLabel(N,Network)

# colormap to plot
my_cmap = cm.get_cmap('rainbow')
my_cmap.set_under('w')

# plot the network cluster
A = label.reshape(N,N)
plt.matshow(A,cmap=my_cmap,vmin=0.0000001)
plt.axis('off')

relabel(label,prp_label,N)
B = label.reshape(N,N)
plt.matshow(B,cmap=my_cmap,vmin=0.0000001)
plt.axis('off')
plt.savefig("HKcluster.pdf",format='pdf' ,bbox_inches='tight')
plt.show()

## Problem 1.3
## Compute the probability P of having a percolating cluster (with different system size) 
#TRY = 100
#P = np.zeros([5,9])
#for L_index in np.arange(5):
#    L = 2**(L_index+2)
#    for p_index in np.arange(9):
#        for trial in np.arange(TRY):
#            p = (p_index+1)*0.1
#            Network = ConstructNetwork(L,p)
#            [label,prp_label] = ClusterLabel(L,Network)
#            relabel(label,prp_label,L)    
#            P[L_index,p_index] = P[L_index,p_index]+spanning(label,L)/TRY
#
## Plot P
#marker_plot = ['^','o','s','d','p']
#for L_index in np.arange(5):
#    L = 2**(L_index+2)
#    plt.plot((np.arange(9)+1)*0.1,P[L_index,:],marker=marker_plot[L_index],markersize=8,markerfacecolor='None',color='black')
#plt.legend(['L=4','L=8','L=16','L=32','L=64'],loc='best')
#plt.xlabel('p')
#plt.ylabel('P(p)')
#plt.savefig("P-p.pdf",format='pdf' ,bbox_inches='tight')
#plt.show()