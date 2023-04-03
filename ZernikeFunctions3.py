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
from numpy import mean, sqrt, square

#def ZernikeDecomposition(rho,phi,m_max,dz,UnitFactor):

def rms(x):
    return np.sqrt(x.dot(x)/x.size)


def ZernikeDecomposition(rho,phi,m_max,dz,UnitFactor):
    A = [[0,0]]
    if (m_max == 4):
        m_max2 = 5
    else:
        m_max2=m_max
    for i in range(1,m_max2):
        nnn=int(np.ceil((-3+np.sqrt(9+8*i))/2.))
        mmm=int(2*i-(nnn*(nnn+2)))
        A.append([mmm,nnn])
    mnlist = ['Z[' + str(A[0][0]) + ']' +'[' + str(A[0][1]) + ']']        
    for i in range(1,len(A)):
        mnlist.append('Z[' + str(A[i][0]) + ']' +'[' + str(A[i][1]) + ']')
    
    ZernikeInfluenceFunctions = np.zeros([len(rho),len(A)])
    for i in range(len(A)):
        
        m = A[i][0]
        n = A[i][1]
        k_inf = int(((n-abs(m))/2))
    
        Zs = np.zeros([len(rho),k_inf+1])
        
        if (abs(m)-n == 0) and (m == 0):
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
        
        if m <= 0:    
            Zs = Zs.reshape(len(Zs))*np.cos(abs(m)*phi)
        else:
            Zs = Zs.reshape(len(Zs))*np.sin(abs(m)*phi)

        ZernikeInfluenceFunctions[:,i] = Zs
        
    # Xlinear = np.dot(np.linalg.pinv(ZernikeInfluenceFunctions),dz)
    Xlinear = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(ZernikeInfluenceFunctions),ZernikeInfluenceFunctions)),np.transpose(ZernikeInfluenceFunctions)),dz)
    Zernikes = Xlinear*ZernikeInfluenceFunctions
    if (m_max == 4):
        ro = [0,1,2,4,3]
        Xlinear[[0,1,2,3,4]]= Xlinear[ro]
        Zernikes[:,[0,1,2,3,4]]= Zernikes[:,ro]
        ZernikeInfluenceFunctions[:,[0,1,2,3,4]]= ZernikeInfluenceFunctions[:,ro]
        mnlist = [val for (_, val) in sorted(zip(ro, mnlist), key=lambda x: x[0])]
        mnlist = mnlist[:-1]
        A = [val for (_, val) in sorted(zip(ro, A), key=lambda x: x[0])]
        A = A[:-1]
        Xlinear=np.delete(Xlinear,4,0)
        Zernikes=np.delete(Zernikes,4,1)
        ZernikeInfluenceFunctions=np.delete(ZernikeInfluenceFunctions,4,1)
    SFEs = np.zeros((m_max))
    for i in range(m_max):
        RMS = np.round(rms(Zernikes[:,i]) * UnitFactor,3)
        SFEs[i] = RMS
    PVs = np.round((np.max(Zernikes,axis=0) - np.min(Zernikes,axis=0)) * UnitFactor,3)
    print('End Zernike Decomp')
    return Zernikes, ZernikeInfluenceFunctions, Xlinear,m,A,SFEs,PVs,mnlist

def CalcZernikeResiduals(rho,phi,dz,UnitFactor,ZernikeNames2):
    A = [[0,0]]

    for i in range(1,85):
        nnn=int(np.ceil((-3+np.sqrt(9+8*i))/2.))
        mmm=int(2*i-(nnn*(nnn+2)))
        A.append([mmm,nnn])
#    mnlist = ['Z[' + str(A[0][0]) + ']' +'[' + str(A[0][1]) + ']']        
#    for i in range(1,len(A)):
#        mnlist.append('Z[' + str(A[i][0]) + ']' +'[' + str(A[i][1]) + ']')
    
    ZernikeInfluenceFunctions = np.zeros([len(rho),len(A)])
    for i in range(len(A)):
        
        m = A[i][0]
        n = A[i][1]
        k_inf = int(((n-abs(m))/2))
    
        Zs = np.zeros([len(rho),k_inf+1])
        
        if (abs(m)-n == 0) and (m == 0):
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
        
        if m <= 0:    
            Zs = Zs.reshape(len(Zs))*np.cos(abs(m)*phi)
        else:
            Zs = Zs.reshape(len(Zs))*np.sin(abs(m)*phi)

        ZernikeInfluenceFunctions[:,i] = Zs
        
    Xlinear = np.dot(np.linalg.pinv(ZernikeInfluenceFunctions),dz)
    Zernikes = Xlinear*ZernikeInfluenceFunctions
    # SFEs = np.zeros((m_max))
    # for i in range(m_max):
    #     RMS = np.round(rms(Zernikes[:,i]) * UnitFactor,3)
    #     SFEs[i] = RMS
    PVs = np.round((np.max(Zernikes,axis=0) - np.min(Zernikes,axis=0)) * UnitFactor,3)
    DZ=dz
    j=0
    standard_seq=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    print('standard')
    print(standard_seq)    
    fringe_seq=[0,1,2,4,3,5,7,8,12,6,9,11,13,17,18,24,10,14,16,19,23,25,31,32,40,15,20,22,26,30,33,39,41,49,50,60,84]
    print('fringe')
    print(fringe_seq)        
    B = [A[i] for i in fringe_seq]
    PVs[standard_seq] = PVs[fringe_seq]
    Zernikes[:,standard_seq] = Zernikes[:,fringe_seq]
    Residuals=np.empty(shape=[23,5],dtype="<U10")
    Residuals[0,:]=['Input Surface','','',str(rms(DZ) * UnitFactor),str(np.round((np.max(DZ) - np.min(DZ)) * UnitFactor,3))]
    PTT=np.empty(shape=[3,1])
    j=1
    for i in range(len(fringe_seq)):
        m = B[i][0]
        n = B[i][1]
        print('i m n',i,m,n)
        if (m ==0 ) and (n == 0):
            DZ=DZ-Zernikes[:,0]
            mag=str(np.format_float_positional( Zernikes[0,0] * UnitFactor ,precision=4))
            phase='0.0'
            resPV=str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS=str(np.round(rms(DZ) * UnitFactor,3))
            print('')
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            Piston = Zernikes[:,0]
            PTT[0,0] = Zernikes[0,0] * UnitFactor
            print('TYPE 1')
            print('I','J','M','N') 
            print(i,j,m,n)
            print(Residuals[:12,:])
            j=j+1
        elif (m == 0) and (n > 0):
            DZ=DZ-Zernikes[:,i]
            mag=str(np.format_float_positional((PVs[i]/2.),precision=4))
            phase='0.0'
            resPV=str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS=str(np.round(np.std(DZ,axis=0) * UnitFactor,3))
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            print('TYPE 2')
            print('I','J','M','N')
            print(i,j,m,n)
            print(Residuals[:12,:])
            j=j+1
#            continue
        elif (m > 0) and (n > 0):
            for k in range(len(fringe_seq)):
                # print('M Match')
                # print(i,k,-(B[k][0]),B[i][0],B[i][1])
                if (-(B[k][0]) == B[i][0]) and (B[k][1] == B[i][1]):
                    l=k
                    print('Match Found',l,B[i][0],B[i][1])
       
            # print('L',l)
            # print('I',i)
            if (i==2):
                Tip = Zernikes[:,i]
                Tilt = Zernikes[l]
                PTT[1,0] = PVs[l]/2
                PTT[2,0] = PVs[i]/2            
            DZ=DZ - Zernikes[:,i] - Zernikes[:,l]
            mag=str( np.format_float_positional(np.sqrt(((PVs[i]/2)**2)+((PVs[l]/2))**2),precision=4  ))
            phase=str(np.arctan2(PVs[i],PVs[l])*180./np.pi)
            resPV=str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS=str(np.round(np.std(DZ,axis=0) * UnitFactor,3))
            print('')
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            print('TYPE 3')
            print('I','J','M','N','L')
            print(i,j,m,n,l)
            print(Residuals[:12,:])
            j=j+1
        # print('HERE')
        #     continue
        # else:
        #     continue
    Zern=Residuals[:,0]   
    Mag=Residuals[:,1]
    Phase=Residuals[:,2]
    ResRMS=Residuals[:,3]
    ResPV=Residuals[:,4]  
    return PTT,Piston,Tip,Tilt,Zern,Mag,Phase,ResRMS,ResPV

#    Zernikes, ZernikeInfluenceFunctions, Xlinear,m,ZernikeModeNames,SFEs,PVs,mnlist,Residuals = ZernikeDecomposition(rho, phi, m_max, dz,UnitFactor, Residuals,ZernikeNames2)    
def ZernikeNamesFunc(m_max):
    ZernikeNames = [' Piston',' Tip',' Tilt',' Astigmatism 1', ' Defocus',' Astigmatism 2',' Trefoil 1',
                    ' Coma 1', ' Coma 2',' Trefoil 2',' ', ' ', ' Spherical Aberration']
    for i in range(1000):
        ZernikeNames.append(' ')
    if (m_max == 4):
        ro = [0,1,2,4,3,5,6,7,8,9,10,11,12]
        ZernikeNames = [val for (_, val) in sorted(zip(ro, ZernikeNames), key=lambda x: x[0])]
    return ZernikeNames    

def ZernikeNamesFunc2():
    ZernikeNames2 = [' Piston',' Tip/Tilt',' Defocus',' Pri Astigmatism',' Pri Coma',
                    ' Pri Spherical', ' PriTrefoil',' Sec Astigmatism',' Sec Coma', ' Sec Spherical', ' Pri Tetrafoil',
                    ' Sec Trefoil',' Ter Astigmatism',' Ter Coma',' Ter Spherical',' Pri Pentafoil',' Sec Tetrafoil',
                    ' Ter Trefoil',' Qua Astigmatism',' Qua Coma',' Qua Spherical',' Quin Spherical']
    for i in range(1000,4):
        ZernikeNames2.append(' ')
    return ZernikeNames2 

def ZernikeTableFunc(mnlist, ZernikeNames, m_max):
    ZernikeTable = []
    ZernikeNames = ZernikeNamesFunc(m_max)
    
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
    file = 'Manual_surface.txt'
    folderfile = folder+file
    
    A = np.loadtxt(folderfile)
#    print(len(A))
#    print(A.size,A.ndim)
    m_max=4
    rho = A[:,0]
    rho = rho/np.max(rho)
#    print(rho)
    phi = A[:,1]
#    print(phi)
    dz =  A[:,2]
    UnitFactor=1E6
#    Residuals=np.empty(shape=[1000,4])
#    Residuals=np.empty(shape=[1000,5],dtype="<U10")
    ZernikeNames = ZernikeNamesFunc(m_max)
    Zernikes, ZernikeInfluenceFunctions, Xlinear,m,ZernikeModeNames,SFEs,PVs,mnlist = ZernikeDecomposition(rho, phi, m_max, dz, UnitFactor)
#   Zernikes, ZernikeInfluenceFunctions, Xlinear,m,ZernikeModeNames,SFEs,PVs,mnlist = ZernikeDecomposition(rho, phi, m_max, data4Zernike,UnitFactor)

    ZernikeTable = ZernikeTableFunc(mnlist, ZernikeNames, m_max)
 
    ZernikeNames2 = ZernikeNamesFunc2()
    Residuals = CalcZernikeResiduals(rho,phi,dz,UnitFactor,ZernikeNames2)            

#    ZernikeTable2 = ZernikeTableFunc2(mnlist, ZernikeNames2, m_max)
    print('Residuals')
    print(Residuals)
#    print('ZernikeTable')
#    print(ZernikeTable)    
    mm = list(range(2,16))
    NN = [3,4,6,10,15,21,28,36,45,55,66,78,91,105]
    for i in range(len(mm)-1):
        NN.append(sum(range(mm[i+1]))) 
#    default_NN = NN.index(6)
    N_Zernikes = 10
    index = NN.index(N_Zernikes)  
    # print('N_Zernikes')
    # print(N_Zernikes)  
    # print('index')
    # print(index)
    # print('m_max')
    # print(m_max)
    # print('NN')     
    # print(NN)
    # print('mm')     
    # print(mm)
    # print('Zernikes')
    # print(Zernikes)
    # print('SFEs')
    # print(SFEs)
    print('PVs')
    print(PVs)
    # print('PVs[1]')
    # print(PVs[1])
    # print('mnlist')
    # print(mnlist)
    # print('ZernikeNames2')
    # print(ZernikeNames2)
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
    for j in range(len(ZernikeModeNames)-1,-1,-1):
        k = np.argsort(SFEs)[-1-j]
        i=j
        print('i',j,i,k)
    print('END')

main()

        
