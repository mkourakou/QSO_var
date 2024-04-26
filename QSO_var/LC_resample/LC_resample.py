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

if not os.path.exists('../Mock_LCs'): 
    os.makedirs('../Mock_LCs')

outpath = '../Mock_LCs/'


#---------------------- Data  --------------------

path = "../LCSIMS/"
filename8_0 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_0.bin"
filename8_5 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_5.bin"
filename8_7 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_7.bin"
filename8_8 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_8.bin"
filename8_9 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_9.bin"

filename9_0 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_0.bin"
filename9_7 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_7.bin"
filename9_5 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_5.bin"
filename9_8 = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+09_ledd1E+00_8.bin"

tabl = Table.read("../LCSIMS/CODE/SIMS/test.fits")
names = [name for name in tabl.colnames if len(tabl[name].shape) <= 1]
df0 = tabl[names].to_pandas()
m = df0['APE_BKG']>0
m = np.logical_and(m, df0['APE_EXP'] > 0)
m = np.logical_and(m, df0['APE_EEF'] > 0)
#m = np.logical_and(m, df0['APE_CTS'] <50 )
m = np.logical_and(m, df0['APE_CTS'] <1000 )
df = df0[m].copy()


# ----------------  Functions  ---------------------



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
    
    
def Sigma2_mod(x, time, tbin) : 
    N = len(time)
    frq = np.arange(1, N/2 + 1).astype(float)/(N*tbin)
    freq = np.copy(frq[frq<=1e-2])
    DFT = fft(x)
    x_mean = np.mean(x)
    P_rms = ((2.0*tbin)/(N*(x_mean**2)))* np.absolute(DFT)**2
    shortP_rms = np.take(P_rms,np.where(frq<=1e-2)[0])
    nu_min = 1/ ( np.max((time-np.min(time))) * 86400 )
    nu_max = 1/(tbin * 86400)
    Sigma2mod = simps(PSD_Model3(freq, LogMBH=8., LogLEDD=0.), freq[freq>nu_min])
    return Sigma2mod  



# ---------------   Short Routine   -------------------


L_i_mbh = [8, 9]
L_i_vers = [0,5,7,8]
L_i_epo = [10, 20, 40, 60, 100, 200]
L_i_cad = [1., 0.5, 0.25, 0.2]

List_LCs = []


bg  = df['APE_BKG'].to_numpy().copy()
texp = df['APE_EXP'].to_numpy().copy() 

for i_mbh in L_i_mbh:
    for i_vers in L_i_vers:
        LC = read_lc(path+"lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+0"+str(i_mbh)+"_ledd1E+00_"+str(i_vers)+".bin")  #dictionary

        tbin = LC['TBIN']  #time bin in seconds 
        N = len(LC['LC'])     # Flux axis of LC
        time = tbin * np.arange(0, N, 1, dtype=int)  #time axis of LC [in seconds]
        x = np.copy(LC['LC'] )
        x[x>3.] = 3.   # set flux upper limit (lightcurves 2, 3, 7 are blown up

        x_mean = np.mean(x)
        X = x/x_mean
        
        np.random.seed(12)
        # LogFluxes_full = np.random.normal(loc=-13.8952, scale=0.4367, size=len(x))

        for i_epo in L_i_epo:
            Number_of_repet = 1000 #1000   #repetition from random starting point of LC
            Number_of_epochs = i_epo  #observation epochs with 6-month cadence

            for i_cad in L_i_cad:
                dt = i_cad * 3.154e+7 #  a yr to seconds
                i_dt = int(dt / tbin) # in time bins  (half a year in 1000s of seconds)
                # print(i_dt, len(LC["LC"])*1000/3.154e+7)
                np.random.seed(12)
                Epochs = np.zeros((Number_of_epochs,Number_of_repet), dtype=int)
                # initial point
                Epochs[0] =  np.sort(np.random.randint(0, high=len(LC["LC"])-Number_of_epochs*i_dt, size=Number_of_repet))
                for iii in range(np.shape(Epochs)[0]-1):
                    Epochs[iii+1] = Epochs[iii] + i_dt

                # NXSV_in = Sigma2_mod( X[Epochs[:,0]], time[Epochs[:,0]], tbin )
                # NXSV_in = Sigma2_mod( X[Epochs[:,0]], time[Epochs[:,0]], i_dt)
                # print(NXSV_in)
                LC_name =  str(i_epo)+"epo_"+str(i_cad)+"yrCad_LC"+str(i_mbh)+"00-"+str(i_vers)  
                #print(LC_name)
                
                PoissonRealCounts = np.zeros((Number_of_epochs,Number_of_repet), dtype=int)
                for i in range(np.shape(Epochs)[1]):   
                    expect = X[Epochs[:,i]]*(1e-13)*texp[0:Number_of_epochs]+bg[0:Number_of_epochs]
                    counts = poisson.rvs(expect)
                    PoissonRealCounts[:,i] = counts

                df_mock = pd.DataFrame()
                SourceCounts = np.zeros(Number_of_repet)
                TotExpo = 0
                for i in range(np.shape(PoissonRealCounts)[0]):
                    df_mock['Counts_Epoch_'+str(i)] = PoissonRealCounts[i,:]
                    df_mock['ExpVal_'+str(i)] = X[Epochs[i,:]]*(1e-13)*texp[i]+bg[i]
                    df_mock['t_exp_'+str(i)] = np.copy(texp[i])
                    df_mock['bgr_'+str(i)] = bg[i]
                    SourceCounts += PoissonRealCounts[i,:]
                    TotExpo += texp[i]

                df_mock['CR_mean'] = SourceCounts/TotExpo
                df_mock.to_csv(outpath+LC_name+".csv", index=False)
                

                


