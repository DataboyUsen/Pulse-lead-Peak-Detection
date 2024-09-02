# -*- coding: utf-8 -*-
"""
This file shows how to generate ppg signals in bulk and multiple labeling methods. 

Note that in this project label refers to the probability of being a peak and actual 
peak information is called peak truth. 

Illustration examples are also included for reference.
@author: SU Xiaochen
"""

# In[Package]

import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import random
from scipy.stats import norm

# In[Labeling Method 1: Naive Method]

def assign_weights(peak_vec,info_vec):
    new_arr = peak_vec.copy()
    for i, x in enumerate(peak_vec):
        if x == 0:
            min_dist = float('inf')
            for j in info_vec:
                dist = abs(i - j)
                min_dist = min(min_dist, dist)
            new_arr[i] = norm.pdf(min_dist, loc=0, scale=4)*(1/norm.pdf(0, loc=0, scale=4))
    return new_arr

# In[Labeling Method 2: Statistical Method]

def get_lable(ppg,alpha=0.3,window_size=30):
    
    
    x=len(ppg)
    threshold=[0]*(window_size//2)
    for j in range(window_size//2,x-window_size//2):
        window=ppg[j-window_size//2:j+window_size//2+1]
        mdmd=np.median(np.abs(ppg-np.median(window)))
        l=len(window)
        element=mdmd*norm.ppf( (1+(1-alpha)**(1/l)) / 2)
        threshold.append(element)
    tail=ppg[(len(ppg) - window_size // 2):]
    threshold.extend(tail)


    fppg=np.where(ppg<=threshold,0,ppg) #filtered ppg
    
    
    sub_datasets = []
    current_sub_dataset = []
    consecutive_zeros = 0
    for x in fppg:
        if x == 0:
            consecutive_zeros += 1
            if consecutive_zeros >= 4:
                if current_sub_dataset:
                    sub_datasets.append(current_sub_dataset)
                    current_sub_dataset = []
                consecutive_zeros = 0
        else:
            consecutive_zeros = 0
            current_sub_dataset.append(x)
    if current_sub_dataset:
        sub_datasets.append(current_sub_dataset)


    standardized_data = []
    for sub_dataset in sub_datasets:
        sub_mean = np.mean(sub_dataset)
        sub_std = np.std(sub_dataset)
        standardized_sub_dataset = [(x - sub_mean) / sub_std for x in sub_dataset]
        standardized_data.extend(standardized_sub_dataset)
        
        
    prob=norm.cdf(standardized_data)

    Pppg = []  #probability of being a peak
    i = 0
    for x in fppg:
        if x == 0:
            Pppg.append(0)
        else:
            Pppg.append(prob[i])
            i += 1

    Pppg = np.nan_to_num(Pppg)
    return Pppg,threshold

# In[Signal Generation]

'''
Note that adjusted  nk.ppg_simulate() is needed for this function. Check adjusted_ppg_simulate.py for more details.
'''

def nppg_geneation(
    sample_rate=60,
    time=5,
    sample_size=5000,
    motion_freq=2,
    motion_amplitude=0.1,
    hr_histgram=False
    ):
    
    signal_length=time*sample_rate
    
    #control randomness
    np.random.seed(567)
    hr = np.round(np.random.normal(90, 14, sample_size*2)).astype(int)
    hr=np.clip(hr, 50, 130)[0:sample_size]
    if hr_histgram:
        plt.hist(hr, bins=30, edgecolor='black')

    random.seed(42)
    signal_seed = random.sample(range(1,7000), sample_size)

    #data processing
    dataset = np.zeros((sample_size, *(signal_length,4)))
    for i in range(sample_size):
        #get signal
        ppg_raw,ppg = nk.ppg_simulate(duration=time, sampling_rate=sample_rate, heart_rate=hr[i], 
                              random_state=signal_seed[i], ibi_randomness=0.3,
                              motion_freq=motion_freq,motion_amplitude=motion_amplitude)
        
        # if use naive method
        peaks, info = nk.ppg_peaks(ppg, sampling_rate=sample_rate, method="elgendi", show=False)
        label=assign_weights(peaks['PPG_Peaks'], info['PPG_Peaks'])
        label=np.where(label<0.05,0,label)
        dataset[i]=np.column_stack((ppg_raw,ppg, label,peaks)) #add 1 data point into dataset
        '''
        # if use statistical method
        label = get_lable(ppg,alpha=0.2,window_size=30)  #statistical threshold gives the Prob. of being a peak
        dataset[i]=np.column_stack((ppg_raw,ppg,label))
        '''  
    return dataset

# In[Illustration Example]

dataset=nppg_geneation(sample_size=10, motion_freq=2, motion_amplitude=1.7)
observation_eg=dataset[0]

def illustration_naive(obervation):
    signal_noise=observation_eg[:,0]
    signal=observation_eg[:,1]
    label=observation_eg[:,2]
    peak_truth=observation_eg[:,3]
    peak_nozero_index=np.nonzero(peak_truth)[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # plot1: signal and signal_noise, showing all peaks
    ax1.plot(signal, color='b', label='Signal')
    ax1.plot(signal_noise, color='r', label='Noisy Signal')
    
    ax1.vlines(peak_nozero_index ,ymin=np.min(signal_noise),ymax=np.max(signal_noise),colors='green', linestyles='dashed', label='Peak by Pulse')
    ax1.legend()
    ax1.set_title('Signal and Noisy Signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # plot2: plot signal_noise,show real peak & estimated peak
    ax2.plot(label, color='r', label='label')
    ax2.vlines(peak_nozero_index ,ymin=np.min(signal_noise),ymax=np.max(signal_noise),colors='green', linestyles='dashed', label='Peak by Pulse')
    ax2.legend()
    ax2.set_title('Peak Truth and Label')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    
    plt.suptitle("Illustration Example(Naive Method)", fontsize=16)
    plt.tight_layout()
    plt.show()
   
illustration_naive(observation_eg)

# In[Details for Statistical Method]

ppg_eg=nk.ppg_simulate(duration=5, sampling_rate=60, heart_rate=90, 
                      random_state=3, ibi_randomness=0.3)[1]

def illustration_stat(ppg_eg):
    prob,threshold=get_lable(ppg_eg,alpha=0.1,window_size=35)
    selected=np.where(ppg_eg>threshold,ppg_eg,0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    ax1.plot(ppg_eg, label='Signal')
    ax1.plot(threshold, label='Dynamic Threshold')
    ax1.legend()
    ax1.set_title("Illustration Example(Statistical Method)")

    ax2.plot(np.where(selected > 0, selected, np.nan), color='green', label='Greater than Threshold')
    ax2.plot(np.where(selected <= 0, selected, np.nan), color='blue', label='Less than or Equal to Threshold')
    ax2.legend()

    ax3.plot(prob, label='prob. of being a peak',color='r')
    ax3.legend()

illustration_stat(ppg_eg)