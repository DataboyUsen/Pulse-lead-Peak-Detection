# -*- coding: utf-8 -*-
"""
Codes regarding recurrence plot generation. 

It starts from examples for signal --> recurrence plot, 
then gives a function used in final program for saving RP and labels in large quantities. 

And a data loader function for reading recurrence plots as grayscale data

@author: SU Xiaochen
"""

# In[Package]

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius #when binary recurrence matrix is required
from pyrqa.metric import EuclideanMetric #when absolute distance is needed in recurrence matrix
from pyrqa.neighbourhood import Unthresholded
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator

import os
import neurokit2 as nk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import Basic_Functions

# In[Example of RP Generation]

'''
Noted that adjusted nk.ppg_simulate() is used
'''

current_path =os.getcwd()
example_path = os.path.join(current_path, "Recurrence Plot(for example)")
os.makedirs(example_path, exist_ok=True)

ppg=ppg_eg=nk.ppg_simulate(duration=5, sampling_rate=60, heart_rate=90, 
                      random_state=3, ibi_randomness=0.3)[1]

#classical RP
time_series = TimeSeries(ppg,
                      embedding_dimension=2,
                      time_delay=1)
settings = Settings(time_series,
                 analysis_type=Classic,
                 neighbourhood=FixedRadius(0.5),
                 similarity_measure=EuclideanMetric,
                 theiler_corrector=1)
computation = RPComputation.create(settings)
result = computation.run()
image_name = f"{example_path}\\Classical_RP.png"
ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    image_name)


#unthreshold RP
time_series = TimeSeries(ppg,
                        embedding_dimension=2,
                        time_delay=1)
settings = Settings(time_series,
                analysis_type=Classic,
                neighbourhood=Unthresholded(),
                similarity_measure=EuclideanMetric,
                theiler_corrector=1)
computation = RPComputation.create(settings)
result = computation.run()
image_name = f"{example_path}\\Unthreshold_RP.png"
ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized,
                                    image_name)   

del time_series, settings, computation, result, image_name, current_path

# In[Function for Saving RPs in Large Quantities]
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
        
# In[Function for Loading RP]

def load_recurrence_and_peak_data(data_dir, file_prefix='obs_', extension='.png', l=10, size=256):
    """
    Load recurrence plot images and their corresponding peak array data.

    Parameters:
    data_dir (str): The directory path where the data files are located.
    file_prefix (str): The prefix used for the recurrence plot image filenames. Default is 'obs_'.
    extension (str): The file extension for the recurrence plot images. Default is '.png'.
    l (int): The number of observations (i.e., the number of recurrence plot images to load). Default is 10.
    size (int): The size of the recurrence plot images (e.g., 128 for 128x128 or 512 for 512x512). Default is 256.

    Returns:
    X (np.ndarray): A NumPy array containing the loaded recurrence plot dataset, with shape (l, size, size, 3),
                    where each image is normalized to the range [0, 1].
    y (np.ndarray): A NumPy array containing the peak array data, loaded from a .npy file.
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


# In[Example of Saving and Loading RPs]

dataset=Basic_Functions.nppg_geneation(sample_size=10, motion_freq=2, motion_amplitude=1.7)

#save the dataset and divide them into 3 sets
save_RP_and_Label(dataset,10,example_path)

#load RP
X_train, y_train = load_recurrence_and_peak_data(example_path, 'train_',l=8,size=299)
X_val, y_val = load_recurrence_and_peak_data(example_path, 'val_',l=1,size=299)
X_test, y_test = load_recurrence_and_peak_data(example_path, 'test_',l=1,size=299)

#plot loaded RP and make comparison
plt.figure(figsize=(8, 8))
plt.imshow(X_train[0])
plt.axis('off')
plt.title('train_0')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(X_val[0])
plt.axis('off')
plt.title('val_0')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(X_test[0])
plt.axis('off')
plt.title('test_0')
plt.show()
