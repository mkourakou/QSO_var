import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math
from scipy.stats import poisson
from scipy.stats import truncnorm, lognorm 
from scipy.integrate import simps, romb
from astropy.table import Table

from scipy.optimize import curve_fit
from scipy.special import gamma
import time as tt
from numpy.fft import fft

import warnings
warnings.filterwarnings('ignore')


#------------------   Data  ------------------------#

path = "../LC_simul/"
filename8_0 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_0.bin"
filename8_5 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_5.bin"
filename8_7 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_7.bin"
filename8_8 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_8.bin"
filename8_9 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_9.bin"

filename9_0 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_0.bin"
filename9_7 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_7.bin"
filename9_5 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_5.bin"
filename9_8 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_8.bin"

tabl = Table.read("../LC_simul/CODE/test.fits")
names = [name for name in tabl.colnames if len(tabl[name].shape) <= 1]
df0 = tabl[names].to_pandas()
m = df0['APE_BKG']>0
m = np.logical_and(m, df0['APE_EXP'] > 0)
m = np.logical_and(m, df0['APE_EEF'] > 0)
#m = np.logical_and(m, df0['APE_CTS'] <50 )
m = np.logical_and(m, df0['APE_CTS'] <1000 )
df = df0[m].copy()

#--------------------   Functions   ---------------------------------------#


def Sigma2_mod(x, time, tbin, lgMBH) : 
    N = len(time)
    frq = np.arange(1, N/2 + 1).astype(float)/(N*tbin)
    freq = np.copy(frq[frq<=1e-2])
    DFT = fft(x)
    x_mean = np.mean(x)
    P_rms = ((2.0*tbin)/(N*(x_mean**2)))* np.absolute(DFT)**2
    shortP_rms = np.take(P_rms,np.where(frq<=1e-2)[0])
    nu_min = 1/ ( np.max((time-np.min(time))) * 86400 )
    nu_max = 1/(tbin * 86400)
    
    Sigma2mod = simps(PSD_Model3(freq, lgMBH, LogLEDD=0.), freq[freq>nu_min])
    return Sigma2mod




def S2_mod( vmin, vmax, LogMBH, LEDD=1.) : 
    A = (3e-3)* (LEDD**(-0.8))
    vb = 580/(10**LogMBH)
    term1 = np.log(vmax/vmin)
    term2 = np.log( (vb+vmax)/(vb+vmin) )
    return A*(term1-term2)





def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def read_lc(filename):
    path1 = os.path.split(filename)
    info = path1[1].split("_")
    TBIN = float(info[4])
    MODEL = int(info[5].split("paolillo")[1])
    MBH =   float(info[7])
    EDD =   float(info[8].split("ledd")[1])
    lc=np.fromfile(filename)
    return {"LC":lc, "TBIN":TBIN, "LGMBH":np.log10(MBH), "LGLEDD": np.log10(EDD), "MODEL": MODEL}

def Plot_lc( x, time):
    plt.close()
    plt.scatter(time, x, color = 'indigo', alpha = 0.2,  marker = '.', s=2, label= 'full LC')
    plt.xlabel('time [sec]')
    plt.ylim(0.2,3. )
    plt.ylabel('Log Flux')
    plt.title(r'Light Curve with LogM_BH = 8 , LogL_Edd = 0, realisation 0')
    plt.grid(alpha=0.3)
    plt.legend( loc = 'upper right')
    return plt.show()

def Plot_lc_ch( x, time, x_subs, time_subs ):
    colorrange = np.arange(0,len(x_subs),1)
    colors = get_colors(colorrange, plt.cm.autumn)
    plt.close()
    plt.scatter(time, x, color = 'indigo', alpha = 0.2,  marker = '.', s=2, label= 'full LC')
    for ii in range(len(x_subs)):
        plt.scatter(time_subs[ii], x_subs[ii], color = colors[ii], alpha = 0.2,  marker = '.', s=2, label= 'LC section'+str(ii))
    
    plt.xlabel('time [sec]')
    plt.ylim(0.2,3. )
    plt.ylabel('Log Flux')
    plt.title(r'Light Curve with LogM_BH = 8 , LogL_Edd = 0, realisation 0')
    plt.grid(alpha=0.3)
    plt.legend( loc = 'center right', bbox_to_anchor=(1.4, 0.5))
    return plt.show()



def PSD_Model2(freq, LogMBH, LogLEDD ):
    
    '''
    PSD Model 2 of Paolillo+17                
    nub=580/(MBH/Msolar)s-1,
    nub*PSD(nub) = 0.02
    # PSD(nub) = A/nub * (1+nub/nub)^-1->
    # nub * PSD(nub) = A / 2 -> A = 2 * nub * PSD(nub)
    '''

    A = np.log10(0.02)
    LogL_bol = LogLEDD + np.log10(1.26)+ 38.0 + LogMBH
    nub = np.log10(200./86400.) + (LogL_bol-44.0) - 2.0*(LogMBH-6.0)
    TMP={"NUB": 10**nub, "PSDNORM": 10**A}
    AA = 10**A
    nu_break = 10**nub
    return AA*(1/freq)*((1+freq/nu_break)**-1)

def PSD_Model3(freq, LogMBH, LogLEDD):
    
    '''
    PSD Model 3 of Paolillo+17                
    nub=580/(MBH/Msolar)s-1,
    nub*PSD(nub) = 3e-3 * LEdd^-0.8
    # PSD(nub) = A/nub * (1+nub/nub)^-1->
    # nub * PSD(nub) = A / 2 -> A = 2 * nub * PSD(nub)
    '''

    LEDD = 10**LogLEDD
    MBH = 10**LogMBH
    AA = (3e-3)* (LEDD**(-0.8))
    nu_break = 580/MBH
    
    return AA*(1/freq)*((1+freq/nu_break)**-1)


def Plot_lc(time, x, save = False):
    color = 'indigo'
    #plt.close()
    plt.scatter(time, x, color = color, alpha = 0.2,  marker = '.', s=2, label= 'LC')
    plt.xlabel('time [sec]')
    plt.ylim(0.2,3. )
    plt.ylabel('Log Flux')
    plt.title(r'Light Curve with LogM_BH = 8 , LogL_Edd = 0, realisation 0')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    return  plt.show()


def Plot_Prms( x, N, tbin, time,  lctype='Full LC'):
    frq = np.arange(1, N/2 + 1).astype(float)/(N*tbin)
    freq = np.copy(frq[frq<=1e-4])
    DFT = fft(x)
    x_mean = np.mean(x)
    P_rms = ((2.0*tbin)/(N*(x_mean**2)))* np.absolute(DFT)**2
    shortP_rms = np.take(P_rms,np.where(frq<=1e-4)[0])
    nu_min = 1/ ( np.max((time-np.min(time))) * 86400 )
    nu_max = 1/(tbin * 86400)
    Sigma2_mod = simps(PSD_Model3(freq, LogMBH=8., LogLEDD=0.), freq[freq>nu_min])
    
    #plt.close()
    plt.scatter(freq, shortP_rms, color='indigo', alpha = 1., label = 'P_rms scatter')
    plt.hexbin(freq, shortP_rms, gridsize = 50, xscale='log', yscale='log', alpha = 0.6, cmap ='Reds') 
    plt.plot(freq, PSD_Model3(freq, LogMBH=8., LogLEDD=0.), color = 'k', label = 'Input PSD (model3)')
    plt.vlines(nu_min, 1e-3, 1e7, linestyles ="dashed", colors ="k")
    plt.vlines(nu_max, 1e-3, 1e7, linestyles ="dashed", colors ="k")
    plt.text(1e-10, 10., '$\sigma^2_{mod} = %.3f$'%( Sigma2_mod ) , fontsize = 16) 
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('P_rms')
    plt.xlim([5e-11,1e-4])
    plt.ylim([1e-3,1e8])
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'PSD of 0-th LC, '+lctype)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower left')
    return plt.show()
    
    
    

# -------------   Read   LCs and load parametres   -----------------------------#


L_i_mbh = [8, 9]
L_i_vers = [0,5,7,8]
L_i_epo = [10, 20, 40, 60, 100, 200]
L_i_cad = [1., 0.5, 0.25, 0.2]

List_LCs = []
List_NXSVs = []

dataframe_collection = {}


for i_mbh in L_i_mbh:
    for i_vers in L_i_vers:
        for i_epo in L_i_epo:
            for i_cad in L_i_cad:
                dt = i_cad * 3.154e+7 #  a yr to seconds
                LC_name =  str(i_epo)+"epo_"+str(i_cad)+"yrCad_LC"+str(i_mbh)+"00-"+str(i_vers)
                List_LCs.append(LC_name)
                LC_file = '../LC_simul/Mock_LCs3/'+LC_name+'.csv'
                dataframe_collection[LC_name] = pd.read_csv(LC_file)






L_i_mbh = [8, 9]
L_i_vers = [0,5,7,8]
L_i_epo = [10, 20, 40, 60, 100, 200]
L_i_cad = [1., 0.5, 0.25, 0.2]

List_LCs = []
List_NXSVs = []

# ASSUMING redshift = 0.
z = 0.
for i_mbh in L_i_mbh:
    for i_vers in L_i_vers:
        for i_epo in L_i_epo:
            for i_cad in L_i_cad:
                LC_name =  str(i_epo)+"epo_"+str(i_cad)+"yrCad_LC"+str(i_mbh)+"00-"+str(i_vers) 
                
                min_time = i_cad * 3.154e+7 #  cadence from yr to seconds
                max_time =  min_time * i_epo   # total duration of the LC
                # print(min_time , max_time)
                vmin = (1+z)/max_time
                vmax = (1+z)/min_time
                # print(vmin<vmax)

                NXSV_in = S2_mod( vmin, vmax, LogMBH=i_mbh, LEDD=1.)
                #print(NXSV_in)
                
                List_LCs.append( LC_name  )
                List_NXSVs.append( NXSV_in )    #SQUARED

dict = {'LCname': List_LCs, 'NXSV_input': List_NXSVs} 
    
df = pd.DataFrame(dict)
    
df
df.to_csv('./Input_NXSVs.csv', index=False)

# print(df[0:10])

# print('\n')
# print('\n')