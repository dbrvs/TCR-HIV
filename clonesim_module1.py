#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# D Reeves, March 2025

## Code to run TCR and HIV simulations

import numpy as np
import scipy.optimize as opt #for power law fitting
import ra_module #my other code
import pandas as pd
import scipy.stats as st

np.seterr(divide = 'ignore') #to deal with spitting out pink boxes??

#function to get resampled proportional abundance (or not in case you want raw pa)
def get_rsa(raw_abundance,resample_flg,sample_size):
    
    a    = -np.sort(-raw_abundance[raw_abundance>0]) #sort and get nonzero
    maxr = len(a[a>1]) #fit to non singletons? 

    if resample_flg:
        rsa = np.random.multinomial(n=sample_size,pvals=a/np.sum(a)) #resampled abundance
        pa = -np.sort(-rsa[rsa>0])/np.sum(rsa) #sort and get nonzero
    
        if np.sum(rsa)!=sample_size:
            print('weird resample!')
    else:
        pa=a/np.sum(a)
        
    return a, pa, maxr #else just raw
    
#note takes N so you can do normalization (even after passing pa)
def calc_ecol(pars,N):
    R=len(pars);  
    D1=np.exp(-np.sum(pars*np.log(pars))) 
    D2=1/np.sum(pars**2)
    
    return R/N,D1/N,D2/N

#runs the whole code
def run_full(modeli):
    
    TCRss=int(1e4) #TCR sample size
    HIVss=int(100) #HIV sample size

    tpts = [10,25,90] #essentially in months (given dt),
    
    #run other functions below
    t,T = sim_clones(modeli) 
    H,T,il = simHIV(t,T,modeli)

    i_H,i_int,i_surv=il #list of indices

    #start the scoring!!
    #CD4
    Trescale = 800 #to get to uL
    CD4 = np.nansum(T[:,0])/Trescale

    #TCRalpha
    al_TCR=[]; d_TCR=[]
    for iit in tpts:

        a,pa,maxr = get_rsa(raw_abundance = T[:,iit],resample_flg = True,sample_size = TCRss) #use resampling code
        r   = np.arange(len(pa))+1 #ranks        
        cpa = np.cumsum(pa) #cumulative prop abundance
        
        d0T,d1T,d2T = calc_ecol(pa,TCRss) #and ecology with other function
        
        d_TCR+=[d0T,d1T,d2T]
        
        #calculate error for distribution, species abundance distribution
        def SAD_error(al1):
            model_pa = r**-al1
            model_pa = model_pa/np.sum(model_pa)
             
            RMS = np.sqrt(np.mean((cpa[:maxr]-np.cumsum(model_pa[:maxr]))**2))
            #KS = np.max(cpa[:maxr]-np.cumsum(model_pa[:maxr]))
            return RMS
        
        res = opt.minimize(SAD_error,x0=[0.8],tol=1e-15, 
                               bounds=[[0.01,2.]], method='L-BFGS-B')
        al_TCR.append(res.x[0])
        
    #HIV provirus fitting
    al_H=[]; d_H=[]
    for iit in tpts[1:]:
        
        a,pa,maxr = get_rsa(raw_abundance = H[:,iit],resample_flg = True,sample_size = HIVss) #use resampling code
        r   = np.arange(len(pa))+1 #ranks        
        cpa = np.cumsum(pa) #cumulative prop abundance
        
        d0H,d1H,d2H = calc_ecol(pa,HIVss) #and ecology with other function
               
        d_H+=[d0T,d1T,d2T]
        
        fit_al,fit_score,ll=ra_module.fit_pwl1(num_fits=30,num_replicates=3,dat_abund=a,R=10**6,max_al=2)
        al=fit_al[np.argmin(fit_score[0])]
        al_H.append(al)

    #TCR fold change metrics from t1>t2
    binz=np.arange(0,30,1)+1
    last_x=10

    t1=tpts[1]; t2=tpts[2]

    dfT=pd.DataFrame()
    dfT['t1']=T[:,t1]/np.nansum(T[:,t1])
    dfT['t2']=T[:,t2]/np.nansum(T[:,t2])

    dfT=dfT.dropna()

    rs1 = np.random.multinomial(n=TCRss,pvals=dfT['t1']) #resampled abundance
    rs2 = np.random.multinomial(n=TCRss,pvals=dfT['t2']) #resampled abundance

    #drop zeros
    T1z = rs1[(rs1>0) & (rs2>0)]
    T2z = rs2[(rs1>0) & (rs2>0)]

    ratios=T2z/T1z #size t1 vs change in size

    rho = st.spearmanr(T1z,ratios)[0]

    expands=ratios[ratios>1]
    contracts=1/ratios[ratios<1]

    if len(expands)>1:
        ce,xe = np.histogram(expands,bins=binz,density=True)

        last_xe = np.where(ce>0)[0][-1]+1 #just up to the last nonzero

        if last_xe==1:
            fit_lame=0
        else:
            ce[ce==0]=np.min(ce[ce>0])
            fit_lame, fit_y0 = np.polyfit(xe[1:last_x],np.log(ce[1:last_x]),1)
    else:
        fit_lame=-100

    if len(contracts)>1:
        cc,xc = np.histogram(contracts,bins=binz,density=True)
        last_xc = np.where(cc>0)[0][-1]+1 #just up to the last nonzero

        if last_xc==1:
            fit_lamc=0
        else:
            #deal with nan
            cc[cc==0]=np.min(cc[cc>0])
            fit_lamc, fit_y0 = np.polyfit(xc[1:last_x],np.log(cc[1:last_x]),1)
    else:
        fit_lamc=-100

    #IPDA
    Htot=np.sum(H,axis=0)
    Hint=np.sum(H[i_int],axis=0)
    Hdef=Htot-Hint

    thI = np.polyfit(t/30,np.log(Hint),1)[0]
    thD = np.polyfit(t/30,np.log(Htot),1)[0]

    metrics = al_TCR+al_H+[fit_lame,fit_lamc,thI,thD,rho]+d_TCR+d_H
    
    return metrics
    
#function that solves stochastically using tau-leap method
def sim_clones(model_dict):
    
    dt=model_dict['dt']
    tF=model_dict['tF']
    T0=model_dict['T0']
    model0=model_dict['m0']
    T0_model_param=model_dict['Tmp']
    rate_model=model_dict['rm']
    rate_model_param=model_dict['rmp']
    net_clearance_rate=model_dict['ncr']
    avg_pro_rate=model_dict['apr']
    avg_emerge_rate=model_dict['aer']
    t_redraw=model_dict['trd']    
    
    sim_size0, clone_dist0 = make_initial_clones(model0, T0, T0_model_param) #use function to initialize the distribution
    
    xt=clone_dist0; xl=[]; tl=[]
    t=0
            
    #make the initial rate vectors
    proli_rate_vector, death_rate_vector = make_vectors(
                rate_model, sim_size0, rate_model_param, net_clearance_rate, avg_pro_rate, avg_emerge_rate/T0)

    ir=1 #index for updating rates
    
    #loop over entire time 
    pois_ok=True
    while t<tF and pois_ok==True:
        xl.append(xt); tl.append(t); #add states to list

        #reset the proliferation rate vectors every t_newrates days
        #next step is to include adjacency matrix so some clones are connected??
        if t > t_redraw*ir:
            proli_rate_vector, death_rate_vector = make_vectors(
                    rate_model, len(xt), rate_model_param, net_clearance_rate, avg_pro_rate, avg_emerge_rate/T0)
            ir+=1
        
        #safety valve on poisson random?
        if (proli_rate_vector*xt*dt).any()>1e3 or (death_rate_vector*xt*dt).any()>1e3:
            prolis=0
            deaths=0
            pois_ok=False
        else:
            prolis=np.random.poisson(proli_rate_vector*xt*dt) #calculate proliferation events for the ith type
            deaths=np.random.poisson(death_rate_vector*xt*dt) #calculate death events for the ith type
        
        xt=xt+prolis-deaths #compute all updated states
        
        xt[xt<0]=0 #make sure no negative numbers
        
        t=t+dt #update time

        #totally new diversity, new clones
        #this shouldn't contribute to existing clones... but maybe its negligible
        births=np.random.poisson(avg_emerge_rate*dt) #calculate birth events for the ith type
        xt=np.append(xt,np.ones(births))
        
        emerge_proli_vector,emerge_death_vector=make_vectors(
                rate_model, births, rate_model_param, net_clearance_rate, avg_pro_rate, avg_emerge_rate/T0) #new rates for new clones
        
        proli_rate_vector=np.append(proli_rate_vector, emerge_proli_vector) #add on new vector   
        death_rate_vector=np.append(death_rate_vector, emerge_death_vector) #add on new vector   

    #postprocess sim_clones to make it an array
    Lf=len(xt) #the max length is that of the last time point
    ci_t=np.zeros([Lf,len(tl)])
    for i in range(len(tl)):
        ci_t[:len(xl[i]),i]=xl[i]

    return np.array(tl), ci_t
    
#initialize the clone distribution
def make_initial_clones(model0, T0, T0_model_param):
    
    #make initial clone sizes
    if model0=='const':
        clone_dist0=np.ones(int(T0/T0_model_param))*T0_model_param #uniform, all = parameter

    #make initial clone sizes
    if model0=='uni':
        clone_dist0=np.random.uniform(1,T0_model_param,[int(T0/T0_model_param)]) #uniform, all between 1 and rate param

    if model0=='exp':
        random_draws = np.random.exponential(T0_model_param,[T0])        
        clone_dist0=np.round(T0*random_draws/np.sum(random_draws))
        clone_dist0=clone_dist0[clone_dist0>0]

    if model0=='pwl':
        #random_draws = np.random.pareto(T0_model_param,[T0])
        #clone_dist0=np.round(T0*random_draws/np.sum(random_draws)) #power law-ish
        #clone_dist0=clone_dist0[clone_dist0>0]
        #sim_size0=len(clone_dist0)
        
        r=np.arange(1,T0)
        pa=r**-T0_model_param
        pa=pa/np.sum(pa)
        clone_dist0=np.random.multinomial(n=T0,pvals=pa,size=1)[0]
        clone_dist0=clone_dist0[clone_dist0>0]

    if model0=='pwl2':
        r=np.arange(1,T0)
        pa=r**-T0_model_param[0]+T0_model_param[2]*r**-T0_model_param[1]
        pa=pa/np.sum(pa)
        clone_dist0=np.random.multinomial(n=T0,pvals=pa,size=1)[0]
        clone_dist0=clone_dist0[clone_dist0>0]
        
    return len(clone_dist0), clone_dist0
    
#make rate arrays given the type of proliferation
def make_vectors(rate_model, sim_size, rate_model_param, net_clearance_rate, avg_pro_rate, avg_emerge_rate_T0scaled):
    
    #all proliferate the same
    if rate_model=='const':
        proli_rate_vector=np.ones(sim_size)*avg_pro_rate

    #all proliferate from uniform 
    if rate_model=='uni':
        proli_rate_vector=np.random.uniform(0,1,[sim_size])*avg_pro_rate

    #uneven proliferation (exponential)
    elif rate_model=='exp':
        proli_rate_vector=np.random.exponential(rate_model_param,[sim_size])*avg_pro_rate

    #uneven proliferation (powerlaw)
    elif rate_model=='pwl':
        proli_rate_vector=np.random.pareto(rate_model_param,[sim_size])*avg_pro_rate

    #2 types of proliferation, both constant, but with different rates
    elif rate_model=='2const':
        n1=int(sim_size*rate_model_param[0]) #number in first phase -- larger clones
        n2=sim_size-n1
        proli_rate_vector=np.append(np.ones(n1)*avg_pro_rate[0],np.ones(n2)*avg_pro_rate[1])

    #2 types of proliferation, both follow exponential, but with different rates
    elif rate_model=='2exp':
        n1=int(sim_size*rate_model_param[0]) #number in first phase -- larger clones
        n2=sim_size-n1
        proli_rate_vector=np.append(np.random.exponential(rate_model_param[1],[n1])*avg_pro_rate[0],
                                    np.random.exponential(rate_model_param[2],[n2])*avg_pro_rate[1])
        
    death_rate_vector=proli_rate_vector - np.ones(sim_size)*(net_clearance_rate - avg_emerge_rate_T0scaled) #enforce balance of birth and death on clone level
        
    return  proli_rate_vector, death_rate_vector
    
#do simulation of HIV clones/proviruses among TCR
#or have another list that just says for each TCR index is it type: 1,2,3,4
#where index tells you if has HIV, intact, survival
#that's getting to be like object oriented!
def simHIV(t,T,modeli):

    decay_params=modeli['pds']
    HIVfullycover=modeli['cov']
    cHIV,cHIVint,cHIVsurv=modeli['pps']
    
    #make full copy of TCR big list, just zero out indices without HIV

    H=np.zeros_like(T) #abudance array
    fH = np.zeros(len(T)) #fraction array

    iTCRl=[]; iTCRlint=[]; iTCRlsurv=[] #indices for various types of clones

    #now loop through and define which TCR have HIV in them
    c=0
    Tnz = np.where(T[:,0]>0)[0] #indices of nonzer initial
    while c<cHIV: #with a little wiggle room?

        iTCR=np.random.choice(Tnz) #pick random one
        #iTCR=np.random.randint(len(T)) #pick random one
        if iTCR in iTCRl:
            iTCR=np.random.choice(Tnz) #try again random one
            
        iTCRl.append(iTCR) #add it to list
            
        Ti=T[iTCR,0] #get that TCR clone size

        #pick the % of cells in taht TCR clone that are HIV infected (can start with all)?
        if HIVfullycover==1:
            Hi = Ti
        else: #if proportional cover pick integers or set equal to avoid randint error
            if Ti>=1:
                Hi = np.random.randint(0,Ti)
            else:
                Hi = Ti
                #HIVc = np.random.poisson(cHIV*TCRclonesize/T0)

        #fH[iTCR] = #need to deal with zero sized TCR clones!!
        fH[iTCR] = Hi/Ti #should be 1 unless not fully covered
        H[iTCR,0] = Hi

        c += Hi #update total count

        #note which one ares intact
        if np.random.rand()<cHIVint/cHIV:
            iTCRlint.append(iTCR)

        #note which one have survival advantages, don't have to be intact, but note can be
        if np.random.rand()<cHIVsurv/cHIV:
            iTCRlsurv.append(iTCR)

    #now do calculation over time for proviruses
    #for now assume HIV stays constant fraction -- if decays extra that just gets added additionally

    #basically now H = POIS(fH*T_t*exp(-xi*t))
    #we solve dT/dt = (a-d)T for T(t) stochastically
    #then we had dH/dt = (a-d-r+s)H = (a-d)H+(-r+s)H
    #so then mean solution is H = H0*exp((a-d)t)*exp((-r+s)t)
    #and we instead use first term as from T cells: H = fH(0)*T(t)*exp((-r+s)*t)

    #T = T + H
    #fH = H/T
    #H = fH*T*exp(t)

    xi_int,xi_def,zeta = decay_params 
    
    #so then loop through and make sure exponent is correct for the H term
    thi=0
    for i in iTCRl:
        thi=-xi_def #everyone gets this
        if i in iTCRlint:
            thi=thi - (xi_int-xi_def) #shouldn't also have defective
        if i in iTCRlsurv:
            thi=thi+zeta
        pois_arg = fH[i]*T[i,:]*np.exp(thi*t)
        #print(fH[i],T[i,:],np.exp(thi*t))
        
        #if pois_arg<1000:
        #    H[i,:]=np.random.poisson(pois_arg)
        #else:
        H[i,:]=np.round(pois_arg)

    H[H<0]=0 #??

    #deal with HIV clones changing numbers of T cells??
    for i in range(len(T)):
        T[i,:]=T[i,:] + H[i,:]
        #T[i,:]=T[i,:] + H[i,0] - H[i,:]

    T[T<0]=0 #is this safe??
    #xt=np.round(xt)??
    
    il = [iTCRl,iTCRlint,iTCRlsurv] #list of indices
    
    return H,T,il


