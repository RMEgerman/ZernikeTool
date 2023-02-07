# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:11:18 2023

@author: egermanrm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma, factorial
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from matplotlib import cm
import streamlit as st
import plotly.graph_objects as go
from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit
import os
from scipy.interpolate import interp1d

#def ZernikeDecomposition(rho,phi,m_max,dz,UnitFactor):



def ZernikeDecomposition(rho,phi,m_max,dz,UnitFactor):
    A = [[0,0]]

    for i in range(1,m_max):
        for j in range(-i,i+1,2):
            A.append([j,i])
    mnlist = ['Z[' + str(A[0][0]) + ']' +'[' + str(A[0][1]) + ']']        
    for i in range(1,len(A)):
        mnlist.append('Z[' + str(A[i][0]) + ']' +'[' + str(A[i][1]) + ']')
    
    ZernikeInfluenceFunctions = np.zeros([len(rho),len(A)])
    print('range(len(A))')
    print(range(len(A)))
    print('A')
    print(A)
    for i in range(len(A)):
        
        m = A[i][0]
        n = A[i][1]
        k_inf = int(((n-abs(m))/2))
    
        Zs = np.zeros([len(rho),k_inf+1])
        
#        print('Zs.size:')
#        print(Zs.size)
#        print('Zs.ndim:')
#        print(Zs.ndim) 
#        print('Zs.shape:')
#        print(Zs.shape)  
        
        if (abs(m)-n == 0) and (m == 0):
            #print('boe')
            k = 0
            F1 = np.math.factorial(n-k)
            F2 = np.math.factorial(k)
            F3 = np.math.factorial(int((n+abs(m))/2) - k )
            F4 = np.math.factorial(int((n-abs(m))/2) - k )
            Zs = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
        else:
            
            for k in range(int((n-abs(m))/2)+1):
                F1 = np.math.factorial(n-k)
                F2 = np.math.factorial(k)
                F3 = np.math.factorial(int((n+abs(m))/2) - k )
                F4 = np.math.factorial(int((n-abs(m))/2) - k )
                Ri = (-1)**k*F1/(F2*F3*F4)*rho**(n-2*k)
                Zs[:,k] = Ri  
            Zs = np.sum(Zs,axis=1)
        
        print('K range')
        print(range(int((n-abs(m))/2)+1),n,m)
        
        if m >= 0:    
            Zs = Zs.reshape(len(Zs))*np.cos(abs(m)*phi)
        else:
            Zs = Zs.reshape(len(Zs))*np.sin(abs(m)*phi)
            
        ZernikeInfluenceFunctions[:,i] = Zs
        
    #Xlinear = np.linalg.lstsq(ZernikeInfluenceFunctions,dz,rcond=None)[0] 
    Xlinear = np.dot(np.linalg.pinv(ZernikeInfluenceFunctions),dz)
    Zernikes = Xlinear*ZernikeInfluenceFunctions
    SFEs = np.round(np.std(Zernikes,axis=0) * UnitFactor,3)
    PVs = np.round((np.max(Zernikes,axis=0) - np.min(Zernikes,axis=0)) * UnitFactor,3)
    return Zernikes, ZernikeInfluenceFunctions, Xlinear,m,A,SFEs,PVs,mnlist

def ZernikeNamesFunc():
    ZernikeNames = [' Piston',' Tip',' Tilt',' Astigmatism 1', ' Defocus',' Astigmatism 2',' Trefoil 1',
                    ' Coma 1', ' Coma 2',' Trefoil 2',' ', ' ', ' Spherical Aberration']
    for i in range(1000):
        ZernikeNames.append(' ')
    return ZernikeNames   

def ZernikeTableFunc(mnlist, ZernikeNames):
    ZernikeTable = []
    ZernikeNames = ZernikeNamesFunc()
    
    for i in range(len(mnlist)):
        ZernikeTable.append(str(mnlist[i])+ZernikeNames[i])
    ZernikeTable.append(' ')
    ZernikeTable.append('Original data:')
    ZernikeTable.append('Quadratic Sum Zernike Terms:')
    ZernikeTable.append('Residual error:')
        
    return ZernikeTable
        
def main():

    print('Spot 1')    

    folder = 'C:/Users/egermanrm/OneDrive - TNO/Zernike_JdV_Testing/'
    file = 'Manual_surface.csv'
    folderfile = folder+file
    
    A = np.loadtxt(folderfile)
#    print(len(A))
#    print(A.size,A.ndim)
    m_max=10
    rho = A[:,0]
    rho = rho/np.max(rho)
#    print(rho)
    phi = A[:,1]
#    print(phi)
    dz =  A[:,2]
    UnitFactor=1
    Zernikes, ZernikeInfluenceFunctions, Xlinear,m,ZernikeModeNames,SFEs,PVs,mnlist = ZernikeDecomposition(rho, phi, m_max, dz,UnitFactor)    
    ZernikeNames = ZernikeNamesFunc()
    ZernikeTable = ZernikeTableFunc(mnlist, ZernikeNames)
    
#    print('ZernikeTable')
#    print(ZernikeTable)    
    mm = list(range(2,16))
    NN = [3,4,6,10,15,21,28,36,45,55,66,78,91,105]
    for i in range(len(mm)-1):
        NN.append(sum(range(mm[i+1]))) 
#    default_NN = NN.index(6)
    N_Zernikes = 10
    index = NN.index(N_Zernikes)  
    m_max = mm[index]
    print('N_Zernikes')
    print(N_Zernikes)  
    print('index')
    print(index)
    print('m_max')
    print(m_max)
    print('NN')     
    print(NN)
    print('mm')     
    print(mm)
#    print('Zernikes.shape')
#    print(Zernikes.shape)
#    print('Zernikes')
#    print(Zernikes)
#    print('Spot 2')
#    print(ZernikeInfluenceFunctions[:,1])
#    print('PVs')
#    print(PVs)
#    print('mm')
    mm = list(range(2,16))
#    print(mm)
    print('END')

main()
