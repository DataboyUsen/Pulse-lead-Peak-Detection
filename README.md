# Pulse-lead Peak Detection

![image](https://github.com/user-attachments/assets/46d78e9c-c84d-4c80-b4e3-b1d7f16eb5d8)



## A. Package Installation: 
#### neurokit2 for signal generation and peak labeling 
    pip install neurokit2 
#### PyRQA for recurrence plot generation and saving 
    pip install PyRQA  
#### tensorflow for CNN model 
    pip install tensorflow 
    pip install pillow 
#### PIL for loading recurrence plot as digital data 
#### seaborn for heat map (not a core package, only for grid search) 
    pip install seaborn 
## B. Program Introduction 

#### • signal_generation_and_labeling.py 
This file shows how to generate ppg signals in bulk and multiple labeling methods. Note 
that in this project label refers to the probability of being a peak and actual peak 
information is called peak truth. Illustration examples are also included for reference. 
#### • RP_stuff.py 
Codes regarding recurrence plot generation. It starts from examples for signal --> 
recurrence plot, then gives a function used in final program for saving RP and labels in 
large quantities. And a data loader function for reading recurrence plots as grayscale data 
#### • adjusted_ppg_simulate.py 
An adjusted version of ppg_simulate( ) function I wrote for adding noise, adjusted 
version is very similar to original one from neurokit2 package except for an extra 
parameter controlling the degree of noise. And adjusted version outputs 2 signals, one is 
normal ppg signal and another is noise-added ppg signal. 
#### • Basic_Functions.py 
Includes functions for multiple purposes. This is a collection for all functions I wrote,  
functions mentioned in other modules are also collected 
#### • feasibility_test.py 
An example of creating data and training a model and calculating relevant evaluation 
standards 
#### • GridSearch.py 
The main part of this project, generating data with different noise level and training one 
model and calculate the distance error matrix & error rate distance


## C. Flowsheet
![flowsheet](https://github.com/user-attachments/assets/9bb7f432-3c76-41a1-869a-ee7f8a11f01f)

