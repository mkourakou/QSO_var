
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
import ultranest
import ultranest.stepsampler
from scipy.optimize import curve_fit
from scipy.special import gamma
import time as tm
from numpy.fft import fft

import warnings
warnings.filterwarnings('ignore')


if not os.path.exists('../NXSV_plots'): 
    os.makedirs('../NXSV_plots')

plotoutpath = '../NXSV_plots/'


if not os.path.exists('../output'): 
    os.makedirs('../output')

outpath = '../output/'


#------------------ Functions  ---------------------

path = "../LCSIMS/Mock_LC3/"
#filename = "lc_lognormpdf_24_tbin_1E+03_paolillo3_mbh_1E+08_ledd1E+00_0.bin"

ECF =  1.104e12


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
    
    
def Likelihood_partial(N, sigma_intr, t, CR_mean, exp_val=None):
    factor = (1/math.factorial(N))*(1/np.sqrt(2*np.pi))*t/sigma_intr
    integrant = (exp_val**N)*np.exp(-exp_val)*np.exp( (-0.5/sigma_intr/sigma_intr)*(exp_val-CR_mean*t)**2)
    I = simps(integrant, exp_val)
    return I*factor 


#--------------------    Load   LCs  ----------------


L_i_mbh = [8, 9]
L_i_vers = [0,5,7,8]
L_i_epo = [10, 20, 40, 60, 100, 200]
L_i_cad = [1., 0.5, 0.25, 0.2]
LC_names = []

dataframe_collection = {}


for i_mbh in L_i_mbh:
    for i_vers in L_i_vers:
        for i_epo in L_i_epo:
            for i_cad in L_i_cad:
                
                LC_name =  str(i_epo)+"epo_"+str(i_cad)+"yrCad_LC"+str(i_mbh)+"00-"+str(i_vers)
                LC_names.append(LC_name)
                LC_file = './Mock_LCs3/'+LC_name+'.csv'
                dataframe_collection[LC_name] = pd.read_csv(LC_file)
        

L_keys = list(dataframe_collection.keys() )



#------------------------------  UltraNest Sample for all LCs  --------------------------#



#limits
mu_lims = [-4.0, 0.]  #LOG Mean Count Rate
sigma_lims = [ -6., 1.5]  #LOG sigma        #sigma_lims = [0.01, 10] 

#prior
def prior_transform(cube):
    params = cube.copy()
    params[0] = (cube[0] * (mu_lims[1] - mu_lims[0]) + mu_lims[0])
    params[1] = (cube[1] * (sigma_lims[1] - sigma_lims[0]) + sigma_lims[0])    #sigma^2
    return params

param_names = ['Log_CRmean', 'Log_NormXSV']



# Posterior sampling
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
LargeN = 1000

ECF =  1.104e12 

# Relevant data
for jj in L_keys[0:40]:
    Ultra_samples = {}
    df = dataframe_collection[jj].copy()
    Number_of_epochs = int((len(df.columns)-1)/4)
    #print(Number_of_epochs)
    
    Obs_Counts = []
    for i in range(Number_of_epochs): Obs_Counts.append(df.loc[1]['Counts_Epoch_'+str(i)]);
    Obs_Counts = np.array(Obs_Counts, dtype=int)

    Obs_Time = []
    for i in range(Number_of_epochs): Obs_Time.append(df.loc[1]['t_exp_'+str(i)]);
    Obs_Time = np.array(Obs_Time)/ECF

    Obs_Bg = []
    for i in range(Number_of_epochs): Obs_Bg.append(df.loc[1]['bgr_'+str(i)]);
    Obs_Bg = np.array(Obs_Bg)

    Obs_CR = Obs_Counts/Obs_Time 

    def log_likelihood(params):   
        CR_mean = 10**params[0]
        sigmaTot =np.sqrt((10**params[1]) * (CR_mean**2))
        low = 0.0
        high = CR_mean + (10*sigmaTot)
        Prop_CR =  truncnorm.rvs(a=(low-CR_mean)/sigmaTot, b=(high-CR_mean)/sigmaTot, loc=CR_mean, scale=sigmaTot, size=LargeN)
        
        LikelihoodSource = np.zeros(shape=Number_of_epochs)
        LikelihoodSource.fill(-200.)
        for i in range(Number_of_epochs):
            # Lightcurve point values
            texp = Obs_Time[i]
            Bgr = Obs_Bg[i]
            Ni = Obs_Counts[i]
    
            # Lambda linspace
            Prop_Lambda = Prop_CR*texp + Bgr
            sample = poisson.pmf(Ni, Prop_Lambda) #the poisson quantity in I.S. summation
            #print(CR_mean, sigmaTot, params[0], params[1], Propo_LambdaValues[0:10], sample[0:10])
            #sys.exit()
            S = sample.sum()  # the sum in I.S. formula
            I = S/LargeN    # I.S. estimation of integral
            if I>=0: 
                LikelihoodSource[i]= np.log(I)    # log likelihood of each lightcurve point
        
            
        TotLsource =  np.sum(LikelihoodSource) 
        return TotLsource

    print("\n --------------- Sampling new LC -----------------\n ")
    start = tm.time()
    sampler = ultranest.ReactiveNestedSampler(param_names,log_likelihood, prior_transform)
    #nsteps = 2 * len(param_names)
    # create step sampler:
    #sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=nsteps,
    #                                                          generate_direction=ultranest.stepsampler.generate_mixture_random_direction )
        
    result = sampler.run(min_num_live_points=1000)
    sampler.print_results()
    ultra_time = tm.time() - start
    print("\n Completed in ", ultra_time, "seconds")
    Ultra_samples['CRmean'] = sampler.results['samples'][:,0]
    Ultra_samples['NXSV'] = sampler.results['samples'][:,1]
    
    pd.DataFrame.from_dict(Ultra_samples).to_csv(outpath+'Output_Ultra50_'+jj+'.csv', index=False)
  



