#!/usr/bin/env python
#used to fit powerlaw distributions by simulation

import matplotlib.pyplot as plt
import numpy as np

#inputs:
#num_replicates, how many replicate simulations/fits
#sim_pa, model array of frequencies, already ranked
#dat_pa, data array of frequencies, already ranked
#sample size, resampling depth to equate model with data
def scorer(dat_abund,num_replicates,sim_pa):#,sample_size):

    score=[]
    sample_size=np.sum(dat_abund)
    for i in range(num_replicates):
        
        sim_abund=sampler(sample_size,sim_pa)
        #dat_abund=sampler(sample_size,dat_pa)
        
        #if len(dat_abund):
        #    dat_abund=np.random.multinomial(sample_size,pvals=sim_pa)
        #    dat_abund=-np.sort(-dat_abund[dat_abund>0])
        
        #make lengths the same, convert back to frequencies
        sub_mat=np.zeros([2,max(len(dat_abund),len(sim_abund))])
        sub_mat[0,:len(dat_abund)]=dat_abund/np.sum(dat_abund)
        sub_mat[1,:len(sim_abund)]=sim_abund/np.sum(sim_abund)

        #print(sub_mat)
        #calculate proportional cumulative abundances
        dat_cpa=np.cumsum(sub_mat[0,:]);
        sim_cpa=np.cumsum(sub_mat[1,:]);
        
        score.append(np.sqrt(np.sum((dat_cpa-sim_cpa)**2))) #rms score from cpas
        #rms_score=max(abs(data_cpa-sample_cpa)); %KS score from cpas
    
    #plt.scatter(dat_cpa,sim_cpa,label=i)
    #plt.legend()    
    return np.mean(score), np.std(score)

#likelihood calculation
def lik(dat_abund,sim_pa):
    
    sim_pa=sim_pa[:len(dat_abund)]/sum(sim_pa[:len(dat_abund)]) #renormalized to correct length
    l=np.sum(dat_abund*np.log(sim_pa))
    
    return l


#do the multinomial sampling
def sampler(sample_size,pa):
    sample=np.random.multinomial(n=sample_size,pvals=pa) #multinomial
    abund=sample[sample>0] #collapse zeros
    #abund= #re-sort
    return -np.sort(-abund)
    
#optimize the single power-law fitter
def fit_pwl1(num_fits,num_replicates,dat_abund,R,max_al):
    
    #R=1e6 #true richness
    al=np.linspace(0.01,max_al,num_fits) #list of parameters to try
    r=np.arange(1,R+1) #model ranks, up to true richness of model
    
    fit_list=[[],[]]; l_list=[]
    for ali in al:
        sim_pa=r**-ali/np.sum(r**-ali)
        
        rms_avg, rms_std = scorer(dat_abund,num_replicates,sim_pa)#,sample_size):

        fit_list[0].append(rms_avg)
        fit_list[1].append(rms_std)
        
        l_list.append(lik(dat_abund,sim_pa))
        
    return al,fit_list,l_list
        
#optimize the double power-law fitter
def fit_pwl2(num_fits,num_replicates,dat_abund,R,max_al):
    
    #R=1e6 #true richness
    al=np.linspace(0.1,max_al,num_fits) #list of parameters to try
    r=np.arange(1,R+1) #model ranks, up to true richness of model
    
    fit_list=[[],[]]
    for ali in al:
        sim_pa=r**-ali/np.sum(r**-ali)
        
        rms_avg, rms_std = scorer(dat_abund,num_replicates,sim_pa)#,sample_size):

        fit_list[0].append(rms_avg)
        fit_list[1].append(rms_std)
        
    return al,fit_list