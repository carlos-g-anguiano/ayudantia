# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:08:34 2021
Updated on Sat Nov  6 14:43:69 2021
@authors: H. Crotte, N. Tepec, D. Lopez, G. Ayala
"""

#Sergey reweighting process' script

import numpy as np
import matplotlib.pyplot as plt
import plot_tools
import ks_test
import scipy.interpolate as interpolate
import seaborn as sns
import os
import pandas as pd
import pdb

def Sergey_rew(data, mc, variables, 
               weightsRD=None, 
               weights=None, 
               ks_threshold=0.5, 
               low_percentile=0.1, 
               high_percentile=99.9,
               bins = 50,
               verbose=True,
               dir_name='gif',
               max_iter = 200,
               make_plots=True
               ):
    
    """ A function based on Sergey's reweighting process to reweight a 
        Monte Carlo distribution. It gets the correct weights by comparing the
        Real Data sample with the MC reweighted sample via the Kolmogorov-Smirnov
        test for each variable on each iteration. Before you start this, make sure
        to select only the elements that have a finite value on each variable of 
        the MC sample. This can be done via the next command:    
        mc=mc[np.isfinite(mc.variable_name)]
    
    Parameters
    ---------------
    data: pd.DataFrame
        Target distribution
    mc: pd.DataFrame
        Monte Carlo distribution. The one that will be reweighted.
    variables: np.array
        An array with the list of desired variables to work with
    weightsRD: np.array, optional
        A list of weights for the real data distribution (default is None)
    weights: np.array, optional
        A list of weights for the MC distribution. Just in case there are. (default is None)
    ks_threshold: float, optional
        A limit for the Kolmogorov Smirnov test (default is 0.5)
    bins: int, dict, optional
        Number of bins. In case of dict, number of bins per variable as key. If any key not in dictionary, the default is set to 50.
    Returns
    ---------------
    total_weights: np.array
        A list with the total weights per event obtained
    ks_test: np.array
        A set of ks-tests for each variable obtained each five iterations. 
        It can be plotted to see the ks-test evolution in function of iterations. 
    """
    
    
    
    alpha_ini= 1/len(variables) 
    ALPHAS   = list()
    alpha    = alpha_ini
    ks_tests = [[] for i in range(len(variables))] #Here is best to use length of variables

    if not type(weightsRD) in [np.ndarray, pd.Series]:
        if weightsRD==None:
            weightsRD = np.ones(len(data)) #Here is best to use number of events of RD
    weightsRD = np.array(weightsRD)

    if not type(weights) in [np.ndarray, pd.Series]:
        if weights==None:
            weights = np.ones_like(mc)
            
        weights = np.array(weights, dtype=np.float32)
        total_weights = np.prod(weights, axis=1)
    else:
        total_weights = weights
        weights = np.ones_like(mc)
        weights = np.array(weights, dtype=np.float32)
    
    total_weights_init = total_weights
    data_hs = list()
    
    if type(bins) is not dict:
        nbins = bins
        bins  = dict()
    else:
        nbins=50
    print('nbins : ', nbins)
    for variable in variables:
        _bins = bins.get(variable, nbins)
        #print(variable,
        #      np.percentile(mc[variable], low_percentile),
        #      np.percentile(mc[variable], high_percentile))
        any_neg=True
        while any_neg:
            histo = np.histogram(data[variable], bins=_bins, 
                        range=[np.percentile(mc[variable], low_percentile),
                               np.percentile(mc[variable], high_percentile)], 
                        weights=weightsRD, 
                        density=True)
            any_neg = np.any(histo[0]<0)
            if any_neg:
                #print(histo[0])
                _bins-=1
            if not any_neg or _bins<=20:  
                any_neg = False
                data_hs.append(histo)
    
    
    
    j=-1
    ks_=[0 for i in range(len(variables))]

    while np.min(ks_)<ks_threshold:
        ALPHAS.append(alpha)
        if j>max_iter: 
            print(ks_)
            break
        
        if verbose:
            print('\n\n\n')
            print(j, ks_)

        j+=1
        
        if j>10:
            np_ks_test = np.array(ks_tests)
            #print('-----Enter----')
            if np.any(np.abs(np_ks_test[:,-1]-np_ks_test[:,-2])>0.1):
                alpha = np.clip(alpha-alpha_ini/2, alpha_ini, 1)
            elif np.all(np.abs(np_ks_test[:,-1]-np_ks_test[:,-2])<0.01):
                alpha = np.clip(alpha+alpha_ini/2, alpha_ini, 1)
        
        
        for index,i in enumerate(variables):
            #print('...',i, total_weights)
            #pdb.set_trace()
            ks = ks_test.ks_2samp_weighted(np.array(data[i]), 
                                           np.array(mc[i]), 
                                           weights1=np.array(weightsRD),
                                           weights2=np.array(total_weights))
            if ks[1]>ks_threshold: 
                ks_[index] = ks[1]
                ks_tests[index].append(ks[1])
                if verbose:
                    print(ks[1])
                continue
                
                
            bin_mean    = (data_hs[index][1][1:]+data_hs[index][1][:-1])/2
            w_histo     = np.histogram(mc[i], bins=data_hs[index][1], 
                                      weights=total_weights,
                                      density=True)
            #pdb.set_trace()
            #plt.title(i)
            #plt.plot(bin_mean, w_histo[0])
            #plt.plot(bin_mean, data_hs[index][0])
            #plt.show()
            ratio_alpha = (alpha*data_hs[index][0] + (1-alpha)*w_histo[0]) / w_histo[0]
            #print('Bins: ', data_hs[index][1])
            #print('Ratio:', ratio_alpha)
            clean_mask     = np.isfinite(ratio_alpha)
            ratio_clean    = ratio_alpha[clean_mask]
            bin_mean_clean = bin_mean[clean_mask]    
        
            ratio_x = interpolate.interp1d(bin_mean_clean,
                                  ratio_clean, 
                                  kind='cubic', 
                                  bounds_error=False, 
                                  fill_value='extrapolate')
            

        
            new_w = ratio_x(mc[i])    
            np.clip(new_w, 0, np.percentile(new_w, 99.5),  out=new_w)

            if j%2==0 and verbose:
                plt.title(f'Weight Variable {i} - {j}')
                plt.scatter(mc[i], new_w)
                plt.show()
            
            weights.T[index] = weights.T[index]*new_w
            mask_rd = (data[i]>=data_hs[index][1][0])&(data[i]<=data_hs[index][1][-1])
            mask_mc = (mc[i]>=data_hs[index][1][0])&(mc[i]<=data_hs[index][1][-1])
            _ks = ks_test.ks_2samp_weighted(
                            data[mask_rd][i], 
                            mc[mask_mc][i], 
                            weights1=weightsRD[mask_rd], 
                            weights2=total_weights[mask_mc])
            ks_[index] = ks[1]
            ks_tests[index].append(ks_[index])

        
        total_weights  = np.prod(weights, axis=1)*total_weights_init
        #total_weights *= np.sum(weightsRD)/np.sum(total_weights)
        
        bin_edges = [histo[1] for histo in data_hs]
        if make_plots:
            make_image(data, mc, weightsRD, total_weights, j, bins = bin_edges, variables=variables, dir_name=dir_name, verbose=verbose)
              
        
    #print(ks_)
    ks_tests_dict = dict()
    for indx, var in enumerate(variables):
        ks_tests_dict[var] = ks_tests[indx]

    
    return ks_tests_dict, total_weights
    

def make_image(data, mc, weightsRD, total_weights, j , bins='none', variables=None, dir_name = 'gif', verbose=True):  
    
    """ Makes and saves an image of the variables' plots every five iterations
    
    Parameters
    -----------------
    data: pd.DataFrame
        Target distribution
    mc: pd.DataFrame
        Monte Carlo distribution. The one that will be reweighted.
    weightsRD: np.array
        A list of weights for the real data distribution
    total_weights: np.array
        A list of total weights for the MC distribution obtained as the product
        of the individual weights previously calculated on the algorithm.
    j: int
        The number of iteration
    
    Returns
    -----------------
    
    """

    
    os.makedirs(dir_name, exist_ok=True)
    data_ = data.copy(deep=True)
    if variables: data_ = data_[variables]


    if j%1==0:            
            est_r  = int(np.floor(np.sqrt(len(data_.T))))   
            est_c  = int(np.ceil(np.sqrt(len(data_.T))))
            if   est_r* est_c<len(data_.T)+1:
                est_c+=1
            fig, axes =plt.subplots(nrows=est_r, ncols=est_c, figsize=(7*est_c, 7*est_r))
            axes = axes.flatten()

            fig.suptitle(f'Iteration {j}')
            for kindx, (var_, column_) in enumerate(data_.items()):

                if bins=='none':
                    bins_=25
                else:
                    bins_=bins[kindx]
                
                #pdb.set_trace()
                axes[kindx].set_title(f'{var_}')
                h = axes[kindx].hist(column_, 
                                     weights=weightsRD,
                                     density=True,
                                     bins=bins_,
                                     label='Real Data',
                                    )
                plot_tools.hist_weighted(mc[var_], 
                                        bins=h[1], 
                                         density=True,
                                        weights=total_weights, 
                                        axis=axes[kindx], 
                                        color='red',
                                        label='MC weighted',
                                        capsize=2, 
                                        verbose=False)
                axes[kindx].set_ylim(0, np.max(h[0])*1.1)
                if kindx==0: axes[kindx].legend(frameon=True, fontsize=15)
                    
            axes[kindx+1].set_title('Total Weights')
            plot_tools.hist_weighted(total_weights, 
                                     bins=50, 
                                     range=[np.min(total_weights),
                                        np.percentile(total_weights, 99)],
                                     color='black',
                                     hist_type='bar',
                                     alpha=0.8,
                                     axis=axes[kindx+1],
                                     verbose=False)
            
            plt.savefig(f'{dir_name}/Iteration{j}.png',  bbox_inches='tight')
            if verbose: plt.show()
            else: plt.close()
            

