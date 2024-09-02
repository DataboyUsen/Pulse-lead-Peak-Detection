# -*- coding: utf-8 -*-
"""
An example of creating data and training a model and calculating relevant evaluation standards
"""

# In[]

import numpy as np
import matplotlib.pyplot as plt


import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf


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


# In[]
n=5000
freq=1.5
amplitude=1.5

#create folders
path=os.path.dirname(os.getcwd())
path=f"{path}\\Feasibility Test"
data_dir=f"{path}\\data"
os.makedirs(path, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)



#generate data & saving RP
dataset=bf.nppg_geneation(sample_rate=60,time=5,sample_size=5000,
                       motion_freq=freq,motion_amplitude=amplitude)
bf.save_RP_and_Label(dataset, n, data_dir)

#model stuff
distance_error,wrong_estimation=model_training(data_dir,path,freq,amplitude,n,300)

peak_truth=np.load(f"{data_dir}\\test_\\peak_truth.npy")
num_of_peaks=np.sum(peak_truth)
mean_distance_error=distance_error/num_of_peaks

print("Mean Distance Error is", mean_distance_error)
print("Error Rate is", wrong_estimation,f"/{n/10}")

#load the model  and testset for prediction
model=tf.keras.models.load_model(f"{path}\\model_Freq{freq}_Amplitude_{amplitude}.h5")
X_test, y_test = bf.load_recurrence_and_peak_data(data_dir, 'test_',l=500,size=299)
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

peak_truth=np.load(f"{path}\\data\\test_\\peak_truth.npy")

bf.estimation_distance(peak_truth,model_result,analysis=True)


#visualization of prediction
bf.estimation_visualization(dataset,model_result,13,peak_truth)
bf.estimation_visualization(dataset,model_result,124,peak_truth)
bf.estimation_visualization(dataset,model_result,472,peak_truth)