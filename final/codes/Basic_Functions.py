# -*- coding: utf-8 -*-
"""
Includes functions for multiple purposes. This is a collection for all functions I wrote, 
functions mentioned in other modules  are also collected
"""

# In[Package]
import numpy as np
import neurokit2 as nk
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
#from pyrqa.neighbourhood import FixedRadius #when binary recurrence matrix is required
from pyrqa.metric import EuclideanMetric #when absolute distance is needed in recurrence matrix
from pyrqa.neighbourhood import Unthresholded
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
import warnings
warnings.filterwarnings("ignore")


import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf



# In[PART 1: about labeling and data generation]







def assign_weights(peak_vec,info_vec):
    '''
    Input 
        2 output results of nk.ppg_peaks() function
    -------
    
    Returns
        label based on naive method
    -------
   
    '''
    new_arr = peak_vec.copy()
    for i, x in enumerate(peak_vec):
        if x == 0:
            min_dist = float('inf')
            for j in info_vec:
                dist = abs(i - j)
                min_dist = min(min_dist, dist)
            new_arr[i] = norm.pdf(min_dist, loc=0, scale=4)*(1/norm.pdf(0, loc=0, scale=4))
    return new_arr













def get_lable(ppg,alpha=0.3,window_size=30):
    '''
    Parameters
    ----------
    ppg : array 
        signal time series
    alpha : int 
        significance level, the lower alpha means selecting more points as peak-related 
        points. 
        The default is 0.3.
    window_size : int
        size for sliding window, usually could be the length of one pulse wave.
        The default is 30.

    Returns
    -------
    Pppg : array
        same length as ppg signal, gives the prob. of being a peak in each corresponding point
    threshold : array
        the dynamic threshold for selecting peak-related points

    '''
    
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



















def nppg_geneation(
    sample_rate=60,
    time=5,
    sample_size=5000,
    motion_freq=2,
    motion_amplitude=0.1,
    hr_histgram=False
    ):
    '''

    generating the dataset for the project     
    
    Parameters
    ----------
    sample_rate : same as samping rate from neurokit2 
    time : same as duration from neurokit2 
    sample_size : number of observations
    motion_freq : controls the number of noise fluctrations 
    motion_amplitude controls the intensity of noise fluctrations 
    hr_histgram : plot the distribution of heart rate or not

    Returns
    -------
    dataset : 
        first column is noisy ppg
        second column is original ppg
        third column is label
        fouth column is peak truth
                
    '''

    
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





















def save_RP_and_Label(dataset,sample_size,path):
    
    n1=int(sample_size*0.8) #number of training set
    n2=int((sample_size-n1)/2) #number of testing set &validation set
    signal_length=len(dataset[0][:,0])
    os.makedirs(f"{path}\\train_", exist_ok=True)
    os.makedirs(f"{path}\\test_", exist_ok=True)
    os.makedirs(f"{path}\\val_", exist_ok=True)
    
    
    for i in range(n1):
        time_series = TimeSeries(dataset[i][:,0],
                             embedding_dimension=2,
                             time_delay=1)
        
        settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=Unthresholded(),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
        
        computation = RPComputation.create(settings)
        result = computation.run()
        image_name = f"{path}\\train_\\train_{i}.png"
        ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized,
                                           image_name)
    
    for i in range(n2):
        time_series = TimeSeries(dataset[n1+i][:,0],
                             embedding_dimension=2,
                             time_delay=1)
        
        settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=Unthresholded(),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
        
        computation = RPComputation.create(settings)
        result = computation.run()
        image_name = f"{path}\\test_\\test_{i}.png"
        ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized,
                                           image_name)
    
    
    for i in range(n2):
        time_series = TimeSeries(dataset[n1+n2+i][:,0],
                             embedding_dimension=2,
                             time_delay=1)
        
        settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=Unthresholded(),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
        
        computation = RPComputation.create(settings)
        result = computation.run()
        image_name = f"{path}\\val_\\val_{i}.png"
        ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized,
                                           image_name)

    #save ground truth
    train_truth = np.zeros((n1, signal_length-1))
    test_truth=np.zeros((n2, signal_length-1))
    val_truth=np.zeros((n2, signal_length-1))
    peak_truth=np.zeros((n2, signal_length-1))

    for i in range(n1):
        train_truth[i]=dataset[i][:,2][:signal_length-1]
    for i in range(n2):
        test_truth[i]=dataset[n1+i][:,2][:signal_length-1]
        val_truth[i]=dataset[n1+n2+i][:,2][:signal_length-1]
        peak_truth[i]=dataset[n1+i][:,3][:signal_length-1]

    if not np.any(np.isnan(train_truth)): 
        np.save(f"{path}\\train_\\train_truth.npy", train_truth)    
    if not np.any(np.isnan(test_truth)): 
        np.save(f"{path}\\test_\\test_truth.npy", test_truth)
    if not np.any(np.isnan(val_truth)): 
        np.save(f"{path}\\val_\\val_truth.npy", val_truth)
        
    if not np.any(np.isnan(peak_truth)): 
        np.save(f"{path}\\test_\\peak_truth.npy", peak_truth)



















def load_recurrence_and_peak_data(data_dir, file_prefix='obs_', extension='.png',l=10,size=256):
    """
    load recurrence plot and peak array data
    
    parametor:
    data_dir (str): data path
    file_prefix (str): prefix of recurrence plot
    extension (str): extension name of recurrence plot
    l(int): number of observations
    size(int): size of recurrence plot(128*128 or maybe 512*512)
    
    output:
    X (np.ndarray): recurrence plot dataset
    y (np.ndarray): peak array
    """
    
    # Get the paths of all recurrence plot image files
    image_files = [os.path.join(f"{data_dir}\\{file_prefix}", f"{file_prefix}{i}{extension}") for i in range(l)]
    
    # Initialize an array to hold the recurrence plot image data
    X = np.zeros((l, size, size, 3), dtype=np.float32)  # Assuming RGB images
    for i, image_file in enumerate(image_files):
        # Open each image file and convert it to a NumPy array, normalizing pixel values to [0, 1]
        image = Image.open(image_file)
        X[i] = np.array(image) / 255.0
    
    # Load the corresponding peak array data from a .npy file
    y = np.load(os.path.join(f"{data_dir}\\{file_prefix}", f"{file_prefix}truth.npy"))
    
    return X, y













# In[PART 2: Model output processing and estimation evaluation]










def peak_estimation(raw_result):
    '''
    

    Parameters
    ----------
    raw_result : single line of model output(raw output for one obervation)
    
    Returns
    -------
    result : binary array
        precise peak position of the observation signal

    '''
    sub_datasets = []
    current_sub_dataset = []
    consecutive_zeros = 0
    for x in raw_result:
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


    selected_data = []
    for sub_dataset in sub_datasets:
        selected_set=np.zeros_like(sub_dataset)
        max_idx = np.argmax(sub_dataset)
        selected_set[max_idx] = 1
        selected_data.extend(selected_set)
       
        
    result=[] 
    i = 0
    for x in raw_result:
        if x == 0:
            result.append(0)
        else:
            result.append(selected_data[i])
            i += 1

    result = np.nan_to_num(result)
    return result















def result_estimation(pred):
    '''
    implement the raw output processing for whole model output
    '''
    final_res=pred.copy()
    
    for i in range(np.shape(pred)[0]):
        final_res[i]=peak_estimation(pred[i])
    
    return final_res










def missing_estimation(truth_vec,est_vec,i):
    '''
    plot real peaks and estimated peaks for i-th obervation 
    '''
    
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(truth_vec, color='green',linewidth=4, label='real')
    ax.plot(est_vec, color='red', linewidth=1.5, label='estimated')

    ax.set_title(f"Situation of Observation {i}")
    ax.set_xlabel("time")
    ax.set_ylabel("is_peak")
    ax.legend()

    return fig














def estimation_distance(truth,est,analysis=False):
    '''
    
    calculate the evaluation standard 


    Parameters
    ----------
    truth : binary array
        peak_truth 
        check the difference of peak_truth and label
    est : birary array
        model estimaiton, raw output isn't  accepted, need output processing 
    analysis : bool
        if yes, plot the difference of bad estimations and print bad observation index

    Returns
    -------
    distance_error : distance between real peak and estimated peak
    wrong_estimation : count of wrong estimation

    '''
    if np.shape(truth)!= np.shape(est): 
        print("Check estimaiton&truth form or shape")
        return(np.nan)
    else:
        distance_error=0
        wrong_estimation=0
        for i in range(np.shape(truth)[0]):
            indexA=np.where(truth[i] == 1)[0]
            indexB=np.where(est[i] == 1)[0]
            if len(indexA)!= len(indexB):
               wrong_estimation+=abs(len(indexA)-len(indexB))
               print("wrong estimation with observation",i)
               if analysis:
                   fig=missing_estimation(truth[i],est[i],i)
                   plt.show()
                   
            else:
                distance_error+=np.sum(abs(indexA-indexB))
                if analysis:
                    if any(num > 8 for num in abs(indexA-indexB)):
                        print("INACCURATE ESTIMATION HAPPENS IN OBERVATION",i)
                        fig2=missing_estimation(truth[i],est[i],i)
                        plt.show()
                    
        return distance_error,wrong_estimation

































def estimation_visualization(data, estimation, index,truth):
    '''
    plot the comparison of clean ppg & noisy ppg for certain observation
    plot the comparison of real peak & estimated peak for certain observation


    Parameters
    ----------
    data : dataset generated by nppg_genatation()
    
    estimation : array
        model output after processing, a matrix with shape [number of observation]*[signal length]
    
    index : int
        index of obervation you want to plot
        
    truth : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    signal = data[index+4000][:, 1] #adjust the index since testing set starts from 4001-st observation of dataset
    signal_noise = data[index+4000][:, 0] #same reason as above
    peak_information = data[index+4000][:, 3]*signal
    peak_nozero_index=np.nonzero(peak_information)[0]
    
    peak_info_not_from_dataset=truth[index]*signal_noise[:299]
    peak_nozero_index_2=np.nonzero(peak_info_not_from_dataset)[0]
    
    
    estimation_peak=estimation[index]*signal_noise[:299]
    estimation_nozero_index=np.nonzero(estimation_peak)[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # plot1: signal and signal_noise, showing all peaks
    ax1.plot(signal, color='b', label='Signal')
    ax1.plot(signal_noise, color='r', label='Signal Noise')
    
    ax1.vlines(peak_nozero_index ,ymin=np.min(signal_noise),ymax=np.max(signal_noise),colors='green', linestyles='dashed', label='Peak by Pulse')
    ax1.legend()
    ax1.set_title('Signal and Signal Noise')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # plot2: plot signal_noise,show real peak & estimated peak
    ax2.plot(signal_noise, color='r', label='Signal_Noise')
    ax2.vlines(peak_nozero_index_2 ,ymin=np.min(signal_noise),ymax=np.max(signal_noise),colors='green', linestyles='dashed', label='Peak by Pulse')
    #ax2.scatter(peak_nozero_index, signal_noise[peak_nozero_index], s=70,marker='^' ,color='green', label='Peak by Pulse')
    ax2.scatter(estimation_nozero_index, signal_noise[estimation_nozero_index], s=70,marker='x' ,color='blue', label='Estimation')
    ax2.legend()
    ax2.set_title('Signal Noise with Peak and Estimation')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    
    
    plt.suptitle(f"Analysis of Observation {index}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    





