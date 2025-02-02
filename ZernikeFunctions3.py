# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 21:23:19 2022

@author: Jan de Vreugd & RM Egerman
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

def rms(x):
    return np.sqrt(x.dot(x)/x.size)

def readme():
    with st.expander('read me'):
        st.write("""
    With this streamlit web-app a Zernike decomposition can be made of sag-data of circular shaped optics. \n
    A data set can be uploaded which contains the x- and y-coordinates and the dz values (sag data). \n
    The data-file should be in .xlsx or .txt format. Rows containing a # may be used as comments and are ignored. \n
    
        """)
        link='The Zernike decomposition is done according to the formulation as described here: [link](https://en.wikipedia.org/wiki/Zernike_polynomials)'
        st.markdown(link,unsafe_allow_html=True)
        
        st.write("""
     The web-app enables substraction of an aspheric curvature from the uploaded data-set. 
     This tends to be applicable for a surface produced from metrology data. \n
     The aspheric curvature is defined according to the following equation:
        """)
        
        st.latex(r'''
                 Z(r) = \frac{Cr^2}{1+\sqrt{1-(1+k)\cdot C^2r^2}}
                 ''')
        st.write(''' where, $C$ is the curvature (inverse radius, $1/R_c$) and $k$ the conical constant. ''')  
        
def dataread(uploaded_file):
        if uploaded_file != 'TestFile_FEM.txt' and uploaded_file != 'TestFile_CMM.txt':
            filename,file_extension = os.path.splitext(uploaded_file.name)
            if file_extension == '.xlsx':
                df = pd.read_excel(uploaded_file , comment="#")
            if file_extension ==  '.txt':
                df = pd.read_csv(uploaded_file, sep = '\s+', header = None , comment="#")
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file, sep = ',', header = None , comment="#")
            shapeFile = df.shape    
            return df, shapeFile
        elif uploaded_file ==  'TestFile_FEM.txt':
            df = pd.read_csv('TestFile_FEM.txt', sep = '\s+', header = None , comment="#")
            shapeFile = df.shape    
            return df, shapeFile
        elif uploaded_file ==  'TestFile_CMM.txt':
            df = pd.read_csv('TestFile_CMM.txt', sep = '\s+', header = None , comment="#")
            shapeFile = df.shape    
            return df, shapeFile

def dataselection(data, shapeFile):
    with st.container():    
        col1,col2,col3 = st.columns(3)
        with col1:
            values = list(range(1,shapeFile[1]+1))
            if shapeFile[1] == 7:
                v = 2
            elif shapeFile[1] == 6:
                v = 1
            elif shapeFile[1] == 4:
                v = 2
            elif shapeFile[1] == 3:
                v = 1   
            else:
                v = 1
            default_ix = values.index(v)
            columnx = st.selectbox('x-column:',values,index = default_ix)
            
        with col2:
            values = list(range(1,shapeFile[1]+1))
            if shapeFile[1] == 7:
                v = 3
            elif shapeFile[1] == 6:
                v = 2
            elif shapeFile[1] == 4:
                v = 3
            elif shapeFile[1] == 3:
                v = 2
            else:
                v = 2
                
            default_iy = values.index(v)
            columny = st.selectbox('y-column:',values,index = default_iy)

        with col3:
            values = list(range(1,shapeFile[1]+1))
            if shapeFile[1] == 7:
                v = 7
            elif shapeFile[1] == 6:
                v = 3
            elif shapeFile[1] == 4:
                v = 4
            elif shapeFile[1] == 3:
                v = 3
            else: 
                v = 3
                
            default_iz = values.index(v)
            columnz = st.selectbox('z-column:',values,index = default_iz) 
            
            x = data.iloc[:,columnx-1].to_numpy()
            x = x.reshape((len(x)))
#            x = x - np.mean(x)        
            y = data.iloc[:,columny-1].to_numpy()
            y = y.reshape((len(y)))
#           y = y - np.mean(y)
            dz = data.iloc[:,columnz-1].to_numpy()
            dz = dz.reshape((len(dz)))
            
            R = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y,x)
            rho = R/np.max(R)
            Rmax = np.max(R)
            return x, y, dz, R, phi, rho , Rmax
        
def SFE_calc(dz,UnitFactor):
    SFE  = np.round(rms(dz)  * UnitFactor,2)
    return SFE

def PV_calc(dz,UnitFactor):
    PV  = np.round( (np.max(dz)-np.min(dz)) * UnitFactor, 2)
    return PV    

def plotlyfunc(x,y,xi,yi,dz,UnitFactor,title):
    
    W = 600
    H = 600
    
    dz_grid = griddata((x,y),dz,(xi,yi),method='cubic')
    
    SFE = str(SFE_calc(dz, UnitFactor))
    PV = str(PV_calc(dz, UnitFactor))
    fig = go.Figure(go.Surface(x=xi,y=yi,z=dz_grid,colorscale='jet'))
    fig.update_layout(title=title + '<br>' + 
                      'PV = ' + PV + 'nm' + '<br>' + 
                      'SFE = ' + SFE + 'nm RMS', autosize=False,width = W, height = H, title_x = 0.5)
    st.plotly_chart(fig, use_container_width=True)
   
# def TipTilt(x,y,dz):
#     A = np.ones((len(x),3))
#     A[:,0] = 1.*A[:,0]
#     A[:,1] = A[:,1] * x 
#     A[:,2] = A[:,2] * y 
    
    #A[:,3] = np.sqrt(A[:,1]**2 + A[:,2]**2)
    
    # Xlinear = np.dot(np.linalg.pinv(A),dz)
    # Xlinear = np.linalg.lstsq(A,dz,rcond=None)[0] 
    # Fit = np.sum(Xlinear*A,axis = 1)
    # ddz = dz - Fit
    # return ddz, Xlinear

def gridarrays(x,y,GridSize):
    X = np.linspace(min(x),max(x),GridSize)
    Y = np.linspace(min(y),max(y),GridSize)
    xi,yi = np.meshgrid(X,Y) # Needs to be checked!!
    return xi,yi
    
def funcSphere(R, Rc, offset):
    C = 1/Rc
    return C*R**2/(1+np.sqrt(1-C**2*R**2)) + offset


def funcASphere(R,Rc,k,offset):
    asphereZ = R**2 / ( Rc * ( 1 + np.sqrt( 1 - (1+k) * (R/Rc)**2  ) ) ) + offset
    return asphereZ

def sagsign(R,dz):
    Ri = np.linspace(np.min(R),np.max(R),100)
    f = interp1d(R, dz)
    dzi = f(Ri)
    ddz = np.diff(dzi)/np.diff(Ri)
    sign = np.sign(sum(ddz))
    return sign

def ZernikeTerms():
    mm = list(range(2,16))
    NN = []
    for i in range(len(mm)-1):
        NN.append(sum(range(mm[i+1])))   
    return NN, mm

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
        
    Xlinear = np.dot(np.linalg.pinv(ZernikeInfluenceFunctions),dz)
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
    # Find Zernike Coefficients: Xlinear
    # Xlinear = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(ZernikeInfluenceFunctions),ZernikeInfluenceFunctions)),np.transpose(ZernikeInfluenceFunctions)),dz)        
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
    Residuals[0,:]=['Input Surface','','',str(np.round(rms(DZ) * UnitFactor,3)),str(np.round((np.max(DZ) - np.min(DZ)) * UnitFactor,3))]
    j=1
    PTT=np.empty(shape=[3,0])
    for i in range(len(fringe_seq)-1):
        m = B[i][0]
        n = B[i][1]
        if (m ==0 ) and (n == 0):
            DZ=DZ-Zernikes[:,0]
            mag=str(np.format_float_positional( Zernikes[0,0] * UnitFactor ,precision=4))
            phase='0.0'
            resPV=str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS=str(np.round(rms(DZ) * UnitFactor,3))
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            Piston = Zernikes[:,0]
            PTT[0,0] = Zernikes[0,0] * UnitFactor
            j=j+1
        elif (m == 0) and (n > 0):
            DZ=DZ-Zernikes[:,i]
            mag=str(np.format_float_positional((PVs[i]/2.),precision=4))
            phase='0.0'
            resPV=str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS=str(np.round(rms(DZ) * UnitFactor,3))
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            j=j+1
        elif (m > 0) and (n > 0):
            for k in range(len(fringe_seq)):

                if (-(B[k][0]) == B[i][0]) and (B[k][1] == B[i][1]):
                    l=k
                    
            if (i==2):
                Tip = Zernikes[:,i]
                Tilt = Zernikes[l]
                PTT[1,0] = PVs[l]/2
                PTT[2,0] = PVs[i]/2 
            DZ = DZ - Zernikes[:,i] - Zernikes[:,l]
            mag = str(np.format_float_positional(np.sqrt(((PVs[i]/2)**2)+((PVs[l]/2))**2),precision=4  ))
            phase = str(np.round((np.arctan2(PVs[i],PVs[l])*180./np.pi),3))
            resPV = str(np.round((np.max(DZ,axis=0) - np.min(DZ,axis=0)) * UnitFactor,3))
            resRMS = str(np.round(rms(DZ) * UnitFactor,3))
            Residuals[j,:]=[ZernikeNames2[j-1],mag,phase,resRMS,resPV]
            j=j+1
        else:
            continue
    Zern=Residuals[:,0]   
    Mag=Residuals[:,1]
    Phase=Residuals[:,2]
    ResRMS=Residuals[:,3]
    ResPV=Residuals[:,4]
    return PTT,Piston,Tip,Tilt,Zern,Mag,Phase,ResRMS,ResPV




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

def ZernikeTableFunc2(mnlist, ZernikeNames, m_max):
    ZernikeTable2 = []
    ZernikeNames2 = ZernikeNamesFunc2()

    for i in range(22):
        ZernikeTable2.append(ZernikeNames2[i])       
    return ZernikeTable2

def PistonTipTiltTableFunc(Xlinear, PTT, PVs, Rmax, UnitFactor, Zernikes):
#   PistonTable = [str(np.format_float_scientific(PTT[0],precision=4))]
    PistonTable = [str(np.format_float_scientific( Zernikes[0,0],precision=4)) ]
#   TipTiltTable = [' ', str(np.format_float_scientific(PTT[1],precision=4)),str(np.format_float_scientific(PTT[2],precision=4))]    
    TipTiltTable = [' ', str(np.format_float_scientific(np.arctan2(PVs[1],2*Rmax*UnitFactor) ,precision=4)),str(np.format_float_scientific( np.arctan2(PVs[2],2*Rmax*UnitFactor)  ,precision=4))]    

    for i in range(1,len(Xlinear)+4):
        PistonTable.append(' ')
    for i in range(3,len(Xlinear)+4):
        TipTiltTable.append(' ')
    return PistonTable, TipTiltTable

def plotly_function(x,y,title):
    
    fsize = 18
    
    fig=go.Figure()
    fig = fig.add_trace(go.Scatter(x=x,y=y,mode = 'markers'))
    fig.update_layout(title_text=title, title_x=0.5,title_font_size=22)
    fig.update_layout(xaxis = dict(tickfont = dict(size=fsize)))
    fig.update_xaxes(title_font_size=fsize)
    fig.update_layout(yaxis = dict(tickfont = dict(size=fsize)))
    fig.update_yaxes(title_font_size=fsize)
    fig.update_layout(width=1000, height=1000)
    fig = fig.update_xaxes(title_text = 'X-coordinates')
    fig = fig.update_yaxes(title_text = 'Y-coordinates')
    fig.update_layout(legend = dict(font = dict(size = fsize, color = "black")))
    st.plotly_chart(fig,use_container_width=False)

# ********** THE STREAMLIT ZERNIKE APP STARTS HERE: ************************


def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title('Zernike Decomposition Tool')
        st.write('info: jan.devreugd@tno.nl')
        readme()
        uploaded_file = st.file_uploader("Select a datafile:")
        
        if uploaded_file is None:
            Testdat_opt = st.checkbox('Use test data')
              
            if Testdat_opt == True:
                Testdataset = st.radio('select data set', ('FE-data', 'Measurement Data'))
                if Testdataset == 'FE-data':
                    uploaded_file = 'TestFile_FEM.txt'
                else:
                    uploaded_file = 'TestFile_CMM.txt'

    if uploaded_file is not None:
               
        data, shapeFile = dataread(uploaded_file)

        with st.sidebar:
            st.write(' \# data points = ' + str(shapeFile[0]) + ', # columns = ' + str(shapeFile[1]) )
            GridSize = st.slider('Select interpolation grid size for 3D plotting', 20, 300, (100))
            units = st.radio('data units:', ('meters', 'millimeters'))
            
        if units == 'meters':
            UnitFactor = 1E9
        else:
            UnitFactor = 1E6
        
        with st.sidebar:
            x,y,dz,R, phi, rho , Rmax= dataselection(data,shapeFile)
            # dzPTT, PTT = TipTilt(x, y, dz)
            ZernikeNames2 = ZernikeNamesFunc2()
            PTT,Piston,Tip,Tilt,Zern,Mag,Phase,ResRMS,ResPV = CalcZernikeResiduals(rho,phi,dz,UnitFactor,ZernikeNames2)            
            dzPTT = dz - Piston - Tip - Tilt
            xi,yi = gridarrays(x,y,GridSize) 

            SphereFit_opt = st.checkbox('Calculate best fitting sphere and asphere')
            Asphere_M = st.checkbox('Subtract Asphere shape from original data')
            if Asphere_M:
                Radius_User = st.number_input('Radius of Curvature:',value = np.max(R)*2,step = 0.001)
                Kappa_User = st.number_input('Conical constant:',step = 0.0001)
                
            ZernikeDecomposition_opt = st.checkbox('Zernike decompostion')
            
            if ZernikeDecomposition_opt:
                NN, mm = ZernikeTerms()
                NN = [3,4,6,10,15,21,28,37,45,55,66,78,91,105]
                default_NN = NN.index(10)
                N_Zernikes = st.selectbox('# Zernike terms: ',NN,index = default_NN)
                index = NN.index(N_Zernikes)  
                m_max = N_Zernikes
                # SortZernikes_opt = st.checkbox('Sort Zernikes',True)
                if (SphereFit_opt==False) and (Asphere_M == False):
                    ZernikeOption = 'Original Data'
                if (SphereFit_opt==True) and (Asphere_M == False):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - Best Fit Sphere','Original Data - Best Fit A-Sphere'))                
                if (Asphere_M==True) and (SphereFit_opt==False):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - UserDefined A-Sphere')) 
                if (SphereFit_opt==True) and (Asphere_M==True):
                    ZernikeOption = st.selectbox('Zernike Decomposition on:',('Original Data', 'Original data - Best Fit Sphere','Original Data - Best Fit A-Sphere','Original data - UserDefined A-Sphere'))    
                    
        with st.expander('Plot original data + Piston Tip Tilt removal:', expanded=True): 
            
            col1, col2 = st.columns(2)
            with col1:
                plotlyfunc(x,y,xi,yi,dz,UnitFactor, 'Original data:')
                plotlyfunc(x,y,xi,yi,dzPTT,UnitFactor,  'Original data minus piston, tip and tilt:')
                if units == 'meters':
                    st.write('piston = ' + str(np.round(1E6*PTT[0],3)) + ' $\mu$m')
                elif units == 'millimeters':
                    st.write('piston = ' + str(np.round(1E3*PTT[0],3)) + ' $\mu$m')
                st.write('tip rotation = ' + str(np.round(1E6*PTT[1],3)) + ' $\mu$rad')
                st.write('tilt rotation = ' + str(np.round(1E6*PTT[2],3)) + ' $\mu$rad')
                                        
        if SphereFit_opt:   
            initial_guess = [sagsign(R,dz)*np.max(R)*10, -1]
            parsS, pcovS = curve_fit(funcSphere, R, dzPTT, p0=initial_guess)
            fitSphere = funcSphere(R,parsS[0],parsS[1])
            dzSphFit =  dzPTT-fitSphere
            
            initial_guess = [sagsign(R,dz)*np.max(R)*10, 0., -1]
            parsAS, pcovAS = curve_fit(funcASphere, R, dzPTT, p0=initial_guess)
            fitASphere = funcASphere(R,parsAS[0],parsAS[1],parsAS[2])
            dzASphFit =  dzPTT-fitASphere
            
            with st.expander('Best sphere and a-sphere fit:'):
                col1, col2 = st.columns(2)
                with col1:
                    plotlyfunc(x,y,xi,yi,dzSphFit,UnitFactor, 'Best fit sphere fit: <br> The best fitting sphere-radius is ' + str(np.round(parsS[0],3)) + ' ' + str(units) )
                with col2:
                    plotlyfunc(x,y,xi,yi,dzASphFit,UnitFactor, 'Best fit Asphere fit: <br> The best fitting Asphere-radius is ' + str(np.round(parsAS[0],3)) + ' ' +  str(units) + 
                               '. <br> The best fitting conical constant is ' + str(np.round(parsAS[1],3)) + '.')
        
        if Asphere_M:
            Asphere_User = funcASphere(R,Radius_User,Kappa_User,0)
            FitAsphereUser = dzPTT - Asphere_User
            FitAsphereUser, PTTAsphereUser = TipTilt(x, y, FitAsphereUser)
            
            with st.expander('Asphere shape Removed:'):
                col1, col2 = st.columns(2)
                with col1:
                    plotlyfunc(x,y,xi,yi,dz,UnitFactor, 'Original data:')
                with col2:
                    plotlyfunc(x,y,xi,yi,FitAsphereUser,UnitFactor, 'Original Data minus Asphere: <br> The selected Asphere-radius is ' + str(np.round(Radius_User,2)) + ' ' +  str(units) + 
                               '. <br> The selected conical constant is ' + str(np.round(Kappa_User,3)) + '.')
                
        if ZernikeDecomposition_opt:
            if ZernikeOption == 'Original Data':
                data4Zernike = dz
            if ZernikeOption == 'Original data - Best Fit Sphere':
                data4Zernike = dzSphFit
            if ZernikeOption == 'Original Data - Best Fit A-Sphere':
                data4Zernike = dzASphFit
            if ZernikeOption == 'Original data - UserDefined A-Sphere':
                data4Zernike = FitAsphereUser
                
            Zernikes, ZernikeInfluenceFunctions, Xlinear,m,ZernikeModeNames,SFEs,PVs,mnlist = ZernikeDecomposition(rho, phi, m_max, data4Zernike,UnitFactor)
            ZernikeNames = ZernikeNamesFunc(m_max)
            ZernikeTable = ZernikeTableFunc(mnlist, ZernikeNames, m_max)
            

            # Piston,Tip,Tilt,Zern,Mag,Phase,ResRMS,ResPV = CalcZernikeResiduals(rho,phi,data4Zernike,UnitFactor,ZernikeNames2)            
            ZernikeTable2 = ZernikeTableFunc2(mnlist, ZernikeNames2, m_max)
       
            with st.expander('Zernike decompostion plots, sorted'):
                col1, col2,col3,col4,col5,col6 = st.columns(6)
                H = [col1,col2,col3,col4,col5,col6]
                
                for j in range(len(ZernikeModeNames)):
                    i=-j
                    plt.figure(i+1)
                    Zjan = griddata((x,y),ZernikeInfluenceFunctions[:,-i],(xi,yi),method='cubic')
                    fig,ax = plt.subplots(figsize=(6,3))
                    pc = ax.pcolormesh(xi,yi,Zjan,cmap=cm.jet)
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_title(ZernikeNames[-i]  + '\n ' + 
                                 'n=' + str(ZernikeModeNames[-i][1]) + ' m=' + str(ZernikeModeNames[-i][0]) + 
                                 '\nPV = ' + str(PVs[-i]) + ' nm' +
                                 '\nSFE = ' + str(SFEs[-i]) + ' nm RMS' 
                                 )
                    with H[j%6]:
                        st.pyplot(fig) 
#
            
            with st.expander('Selected data minus summation of Zernikes'):
                ZernikesSum = np.sum(Zernikes,axis = 1)
                ZernikeDelta = data4Zernike - ZernikesSum
                
                col1, col2 = st.columns(2)
                with col1:
                    plotlyfunc(x,y,xi,yi,data4Zernike,UnitFactor, ZernikeOption)
                with col2:    
                    plotlyfunc(x,y,xi,yi,ZernikeDelta,UnitFactor, '(' + ZernikeOption + ')' + ' minus Zernikes:')                        
            
            with st.expander('Zernike Table'):
                PistonTable, TipTiltTable = PistonTipTiltTableFunc(Xlinear,PTT,PVs,Rmax,UnitFactor,Zernikes)
                SFEColumn = SFEs
                SFEColumn = np.append(SFEColumn, ' ')
                SFEColumn = np.append(SFEColumn,  str(np.round(rms(dz)*UnitFactor,3))    )
                SFEColumn = np.append(SFEColumn,  np.round(np.sum(np.sqrt(np.sum(SFEs**2))),3) )
                SFEColumn = np.append(SFEColumn,  str(np.round(rms(ZernikeDelta)*UnitFactor,3))    )
                
                # PVs[0] = len(mnlist)
                # PVs[1] = m_max
                # PVs[2] = N_Zernikes  
                # PVs[3] = index
                # PVs[4] = default_NN
                PVs = np.append(PVs, ' ')
                PVs = np.append(PVs, str(np.round((np.max(dz) - np.min(dz))*UnitFactor , 3)) )
                PVs = np.append(PVs, ' ' )
                PVs = np.append(PVs, str(np.round((np.max(ZernikeDelta) - np.min(ZernikeDelta))*UnitFactor , 3)) )
                
     
                if units == 'meters':
                    dfTable = pd.DataFrame({'Zernike Mode:' : ZernikeTable, 'PV [nm]' : PVs, 'SFE [nm RMS]:' : SFEColumn, 'Piston [m]:' : PistonTable, 'Tip Tilt angle [rad]:' : TipTiltTable}) 
                elif units == 'millimeters':
                    dfTable = pd.DataFrame({'Zernike Mode:' : ZernikeTable, 'PV [nm]' : PVs, 'SFE [nm RMS]:' : SFEColumn, 'Piston [mm]:' : PistonTable, 'Tip Tilt angle [rad]:' : TipTiltTable}) 
                #st.write(dfTable) 

                st.table(dfTable.style)
             # Zern,Mag,Phase,ResRMS,ResPV = CalcZernikeResiduals(rho,phi,data4Zernike,UnitFactor,ZernikeNames2)            
               
            with st.expander('Zernike Residual Table - Mag/Phase thru Quin Spherical'):
                dfTable2 = pd.DataFrame({'Zernike' : Zern, 'Mag [nm]' : Mag, ' Phase[deg]' : Phase, 'ResRMS[nm]' : ResRMS, 'ResPV[nm]' : ResPV}) 
                st.table(dfTable2.style)
               
                
        with st.expander('X and Y locations of all datapoints'):
            plotly_function(x,y,'data coordinates')
                          
   
main()
