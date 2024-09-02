# -*- coding: utf-8 -*-
"""
The main part of this project, generating data with different noise level and 
training one model and calculate the distance error matrix & error rate distance

"""





# In[0]Package

import numpy as np
import matplotlib.pyplot as plt


import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import seaborn as sns


import Basic_Functions as bf

# In[1]Model Architecture and Packaed Process for Model Training 

def create_model(X_train,y_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    return model


def model_training(data_dir,model_dir,frequency,amplitude,sample_size=5000,signal_length=300):
    n1=int(sample_size*0.8) #number of training set
    n2=int((sample_size-n1)/2) #number of testing set &validation set
    
    X_train, y_train = bf.load_recurrence_and_peak_data(data_dir, 'train_',l=n1,size=signal_length-1)
    X_val, y_val = bf.load_recurrence_and_peak_data(data_dir, 'val_',l=n2,size=signal_length-1)
    X_test, y_test = bf.load_recurrence_and_peak_data(data_dir, 'test_',l=n2,size=signal_length-1)
    
    tf.random.set_seed(42)
    
    model=create_model(X_train,y_train)
    
    model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=20, 
                       batch_size=32)
    
    y_pred = model.predict(X_test)
    y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
    model_result=bf.result_estimation(y_pred_adjust)
    tf.keras.models.save_model(model, f"{model_dir}\\model_Freq{frequency}_Amplitude_{amplitude}.h5")
    
    peak_truth=np.load(f"{data_dir}\\test_\\peak_truth.npy")
    distance_error, wrong_estimation= bf.estimation_distance(peak_truth,model_result)
    return distance_error,wrong_estimation


# In[2]Grid Search
n=5000

path=os.path.dirname(os.getcwd())
path=f"{path}\\GridSearch"
Dpath=f"{path}\\data"
Mpath=f"{path}\\model"


os.makedirs(path, exist_ok=True)
os.makedirs(Dpath, exist_ok=True)
os.makedirs(Mpath, exist_ok=True)



freq_range=np.arange(1.5, 6.1, 0.5)
amplitude_range=np.arange(0.5, 2.4, 0.2)
amplitude_range=np.round(amplitude_range, 1)# adjust the digits of amplitude_range


distance_error_matrix=np.zeros((10,10))
wrong_estimation_matrix=np.zeros((10,10))


for i,freq in enumerate(freq_range):
    for j,amplitude in enumerate(amplitude_range):
        #create folders
        data_dir=f"{Dpath}\\freq{freq:.1f}_amp{amplitude:.1f}"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}\\train_", exist_ok=True)
        os.makedirs(f"{data_dir}\\test_", exist_ok=True)
        os.makedirs(f"{data_dir}\\val_", exist_ok=True)
        
        #generate data & saving RP
        dataset=bf.nppg_geneation(sample_rate=60,time=5,sample_size=5000,
                               motion_freq=freq,motion_amplitude=amplitude)
        bf.save_RP_and_Label(dataset, n, data_dir)
        
        #model stuff
        distance_error,wrong_estimation=model_training(data_dir,Mpath,freq,amplitude,n,300)
        distance_error_matrix[i, j]=distance_error
        wrong_estimation_matrix[i,j]=wrong_estimation

# In[3]Evaluation

# distance error
plt.figure(figsize=(10, 8))
sns.heatmap(distance_error_matrix, fmt='.0f',annot=True, cmap='YlOrRd', xticklabels=amplitude_range, yticklabels=freq_range)
plt.xlabel('amplitude')
plt.ylabel('frequency')
plt.title('GridSearch:Distance Error')
plt.show()

# mean distance error
mean_distance_error=distance_error_matrix.copy()
for i,freq in enumerate(freq_range):
    for j,amplitude in enumerate(amplitude_range):
        data_dir=f"{path}\\freq{freq:.1f}_amp{amplitude:.1f}"
        peak_truth=np.load(f"{data_dir}\\test_\\peak_truth.npy")
        num_of_peaks=np.sum(peak_truth)
        mean_distance_error[i,j]=mean_distance_error[i,j]/num_of_peaks
        
        
plt.figure(figsize=(10, 8))
sns.heatmap(mean_distance_error,annot=True, cmap='YlOrRd', xticklabels=amplitude_range, yticklabels=freq_range)
plt.xlabel('amplitude')
plt.ylabel('frequency')
plt.title('GridSearch:Mean Distance Error')
plt.show()


# wrong estimation
plt.figure(figsize=(10, 8))
sns.heatmap(wrong_estimation_matrix, fmt='.0f',annot=True, cmap='YlOrRd', xticklabels=amplitude_range, yticklabels=freq_range)
plt.xlabel('amplitude')
plt.ylabel('frequency')
plt.title('GridSearch:# of wrong estimation')
plt.show()




# In[4]Analysis on dataset having most noise

freq=6.0
amp=2.3
    

#load the model  and testset for prediction
dataset=bf.nppg_geneation(sample_rate=60,time=5,sample_size=5000, motion_freq=freq,motion_amplitude=amp)

model=tf.keras.models.load_model(f"{Mpath}\\model_Freq{freq}_Amplitude_{amp}.h5")

Dpath_specific = f"{Dpath}\\freq{freq}_amp{amp}"
X_test, y_test = bf.load_recurrence_and_peak_data(Dpath_specific, 'test_',l=500,size=299)
y_pred = model.predict(X_test)



#visulization of raw output
fig, axes = plt.subplots(4, 4, figsize=(30, 20))

for i, ax in enumerate(axes.flat):
    if i < 16:
        ax.plot(y_test[i], color='green',linewidth=4 ,label='ground truth')
        ax.plot(y_pred[i], color='red', linewidth=2, label='estimation')
        ax.set_title(f'Plot {i}')
        ax.legend()

    else:
        ax.axis('off') 

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()




# check case having relatively high distance error or wrong estimation
y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
model_result=bf.result_estimation(y_pred_adjust)
peak_truth=np.load(f"{Dpath_specific}\\test_\\peak_truth.npy")
bf.estimation_distance(peak_truth,model_result,analysis=True)



#visualization of prediction
bf.estimation_visualization(dataset,model_result,90,peak_truth)
bf.estimation_visualization(dataset,model_result,116,peak_truth)
bf.estimation_visualization(dataset,model_result,195,peak_truth)
bf.estimation_visualization(dataset,model_result,224,peak_truth)
