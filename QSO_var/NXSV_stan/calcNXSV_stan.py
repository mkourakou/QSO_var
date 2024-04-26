import os
import sys
import glob
import pickle
import asyncio
import nest_asyncio


from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set()
plt.rcParams["figure.figsize"] = (10, 5)

import pandas as pd

import argparse
# import torch

# import pyro
# import pyro.distributions as dist
# import pyro.infer.mcmc as mcmc
import stan
# from pyro.distributions import Normal, Uniform
# from pyro.infer import EmpiricalMarginal, Importance

from scipy.stats import poisson
from scipy.stats import truncnorm
from scipy.integrate import simps, romb
from scipy.optimize import curve_fit
from scipy.special import gamma
import time as tm

import warnings
warnings.filterwarnings('ignore')

if not os.path.exists('../NXSV_plots'): 
    os.makedirs('../NXSV_plots')

outpath = '../NXSV_plots/'


if not os.path.exists('../output'): 
    os.makedirs('../output')

stanoutpath = '../output/'


#------------------ Functions  ---------------------

def add_plot(x, axis, true, label):
    #sns.distplot(x, ax=axis, axlabel=label)
    axis.axvline(true, color="black", label = 'input value')
    axis.axvline(np.median(x), color="red", label = 'median sampled')
    axis.axvline(np.mean(x), color="green", label = 'mean sampled')


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
                LC_file = '../LC_simul/Mock_LCs3/'+LC_name+'.csv'
                dataframe_collection[LC_name] = pd.read_csv(LC_file)
        

L_keys = list(dataframe_collection.keys() )



#------------------------------  Stan Sample for all LCs  --------------------------#



async def main(): print(1)

try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # 'RuntimeError: There is no current event loop...'
    loop = None

if loop and loop.is_running():
    print('Async event loop already running. Adding coroutine to the event loop.')
    tsk = loop.create_task(main())
    # ^-- https://docs.python.org/3/library/asyncio-task.html#task-object
    # Optionally, a callback function can be executed when the coroutine completes
    tsk.add_done_callback(
        lambda t: print(f'Task done with result={t.result()}  << return val of main()'))
else:
    print('Starting new event loop')
    result = asyncio.run(main())

nest_asyncio.apply()


stan_file = """
data {     //types and dimensions of the data

  int<lower=1> N; // number of objects (length oflightcurve)
  int<lower=0> Counts[N]; // observed counts for object 
  vector[N] Time;
  vector[N] Bg;
  real LowTrunc;
  

  //real Time[N]; // exposure time (not including ECF and EEF) 
  // vector[N] CR;

  //real Bg[N]; // background counts for object
  
 }


parameters {

    real<lower=0., upper=10.>  CR_mean_s; // Mean Count Rate
    real<lower=0., upper=20.> Sigma_tot_s; // Sigma NXSV (Squared!)
    real<lower=0.> CR;
  
}


transformed parameters {
  real CR_mean = CR_mean_s / 100;
  real Sigma_tot = Sigma_tot_s/100;
}


model{

  vector[N] Lambda;

  Lambda = CR * Time + Bg;

  // target += uniform_lpdf(CR_mean_s | 0., 10.);
  // target += uniform_lpdf(Sigma_tot_s | 0., 10.);
  // target += normal_lpdf(CR | CR_mean, Sigma_tot);
  // target += poisson_lpmf(Counts | Lambda);
  
  CR ~ normal(CR_mean, Sigma_tot) T[LowTrunc, ];
  Counts ~ poisson(Lambda);
  

}

"""



NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 20
ECF =  1.104e12 

Stan_NXSVs = {}
Stan_sampled = []
start = tm.time()
# Relevant data
for jj in tqdm(L_keys):
    df = dataframe_collection[jj].copy()
    Number_of_epochs = int((len(df.columns)-1)/4)

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

    stan_data = {'N': Number_of_epochs, 'Counts': Obs_Counts, 
             'Time': Obs_Time, 'Bg': Obs_Bg, 'LowTrunc': 0.0 }

    stan_poster = stan.build(stan_file, data=stan_data)
    stan_fit = stan_poster.sample(num_chains=NUM_CHAINS, num_samples=NUM_WARMUP + NUM_SAMPLES, num_warmup=NUM_WARMUP)
    stan_df = stan_fit.to_frame()
    stan_df['Sigma_NXSV'] = np.sqrt( stan_df['Sigma_tot']*stan_df['Sigma_tot']*stan_df['CR_mean']*stan_df['CR_mean'] )
    stan_df['Sigma_NXSV_cr'] = np.sqrt( stan_df['Sigma_tot']*stan_df['Sigma_tot']*stan_df['CR']*stan_df['CR'] )
    
    
    print("---------------------------------------------------------------------------\n")
    print("----------------          SAMPLING COMPLETED        -----------------------\n")
    print("---------------------------------------------------------------------------\n")
    Stan_NXSVs[jj] = stan_df 
    stan_df.to_csv(stanoutpath+'Output_Stan_'+str(jj)+'.csv', index=False)


stan_time = tm.time() - start

print("\n \n ")
print("STAN time: ", stan_time)
print("\n \n")




#len(Stan_NXSVs.keys())
df_input = pd.read_csv('../inNXSV/Input_NXSVs.csv')
print("We are working on the same LCs: ", df_input['LCname'].to_list()==list(Stan_NXSVs.keys()), "\n \n ")

NXSV_means = []
NXSV_medians = []
NXSV_trues =[]
Sigma_means =[]
Sigma_medians =[]
           
for jj in range(len(list(Stan_NXSVs.keys()))):
    df = Stan_NXSVs[list(Stan_NXSVs.keys())[jj]].copy()
    #print(list(Stan_NXSVs.keys())[jj] == df_input.loc[jj]["LCname"] )
    SamplMean_NXSV = df['Sigma_NXSV'].mean()
    SamplMedian_NXSV = df['Sigma_NXSV'].median()
    True_NXSV = df_input.loc[jj]['NXSV_input']
    NXSV_means.append(SamplMean_NXSV)
    NXSV_medians.append(SamplMedian_NXSV)
    NXSV_trues.append(True_NXSV)
    Sigma_means.append(df['Sigma_tot'].mean())
    Sigma_medians.append(df['Sigma_tot'].median())

    fig, axes = plt.subplots(1, 1)
    add_plot(df["Sigma_NXSV"], axes, True_NXSV, r"$\sigma_{NXSV}$")
    plt.title(list(Stan_NXSVs.keys())[jj]+"  sampled NXSV = "+ str(SamplMean_NXSV))
    plt.legend()
    #plt.savefig(outpath+list(Stan_NXSVs.keys())[jj]+".png")
    plt.show()
    



