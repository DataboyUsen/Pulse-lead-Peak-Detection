# -*- coding: utf-8 -*-
"""
The main part of this project, generating data with different noise level and 
training one model and calculate the distance error matrix & error rate distance

"""





# In[0]Package

import numpy as np
import neurokit2 as nk
import random
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import gamma

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius #when binary recurrence matrix is required
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


from matplotlib import cm
#import seaborn as sns

import Basic_Functions
# In[1]Model Architecture and Packaed Process for Model Training 
    new_arr = peak_vec.copy()
    for i, x in enumerate(peak_vec):
        if x == 0:
            min_dist = float('inf')
            for j in info_vec:
                dist = abs(i - j)
                min_dist = min(min_dist, dist)
            new_arr[i] = norm.pdf(min_dist, loc=0, scale=4)*(1/norm.pdf(0, loc=0, scale=4))
    return new_arr
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
    # 获取所有recurrence plot图像文件路径
    image_files = [os.path.join(f"{data_dir}\\{file_prefix}", f"{file_prefix}{i}{extension}") for i in range(l)]
    
    # 加载recurrence plot图像数据
    X = np.zeros((l, size, size, 3), dtype=np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        X[i] = np.array(image) / 255.0
    
    # 加载对应的peak array数据
    y = np.load(os.path.join(f"{data_dir}\\{file_prefix}", f"{file_prefix}truth.npy"))
    
    return X,
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
    
    X_train, y_train = load_recurrence_and_peak_data(data_dir, 'train_',l=n1,size=signal_length-1)
    X_val, y_val = load_recurrence_and_peak_data(data_dir, 'val_',l=n2,size=signal_length-1)
    X_test, y_test = load_recurrence_and_peak_data(data_dir, 'test_',l=n2,size=signal_length-1)
    
    tf.random.set_seed(42)
    
    model=create_model(X_train,y_train)
    
    model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=20, 
                       batch_size=32)
    
    y_pred = model.predict(X_test)
    y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
    model_result=result_estimation(y_pred_adjust)
    tf.keras.models.save_model(model, f"{model_dir}\\model_Freq{frequency}_Amplitude_{amplitude}.h5")
    
    peak_truth=np.load(f"{data_dir}\\test_\\peak_truth.npy")
    distance_error, wrong_estimation= estimation_distance(peak_truth,model_result)
    return distance_error,wrong_estimation


# In[4]Grid Search
n=5000
path = os.getcwd()
path=os.path.dirname(path)

freq_range=np.arange(1.5, 6.1, 0.5)
amplitude_range=np.arange(0.5, 2.4, 0.2)
amplitude_range=np.round(amplitude_range, 1)# adjust the digits of amplitude_range


distance_error_matrix=np.zeros((10,10))
wrong_estimation_matrix=np.zeros((10,10))


for i,freq in enumerate(freq_range):
    for j,amplitude in enumerate(amplitude_range):
        #create folders
        data_dir=f"{path}\\freq{freq:.1f}_amp{amplitude:.1f}"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}\\train_", exist_ok=True)
        os.makedirs(f"{data_dir}\\test_", exist_ok=True)
        os.makedirs(f"{data_dir}\\val_", exist_ok=True)
        
        #generate data & saving RP
        dataset=nppg_geneation(sample_rate=60,time=5,sample_size=5000,
                               motion_freq=freq,motion_amplitude=amplitude)
        save_RP_and_Label(dataset, n, data_dir)
        
        #model stuff
        model_dir="C:\\Users\\User\\Usen_RQA\\gridsearch\\model"
        distance_error,wrong_estimation=model_training(data_dir,model_dir,freq,amplitude,n,300)
        distance_error_matrix[i, j]=distance_error
        wrong_estimation_matrix[i,j]=wrong_estimation

# In[5]Evaluation

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




# In[6]Analysis on dataset having most noise

dataset=nppg_geneation(sample_rate=60,time=5,sample_size=5000, motion_freq=6.0,motion_amplitude=2.3)



model=tf.keras.models.load_model('C:/Users/User/Usen_RQA/gridsearch/model/model_Freq6.0_Amplitude_2.3.h5')
Dpath = "C:\\Users\\User\\Usen_RQA\\gridsearch\\data\\freq6.0_amp2.3"
X_test, y_test = load_recurrence_and_peak_data(Dpath, 'test_',l=500,size=299)

y_pred = model.predict(X_test)


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




# check case having relatively high distance error

y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
model_result=result_estimation(y_pred_adjust)

peak_truth=np.load(f"{Dpath}\\test_\\peak_truth.npy")

estimation_distance(peak_truth,model_result,analysis=True)



# 90 116 195 224 382 478 496 have problems
estimation_visualization(dataset,model_result,90)
estimation_visualization(dataset,model_result,116)
estimation_visualization(dataset,model_result,195)
estimation_visualization(dataset,model_result,224)
estimation_visualization(dataset,model_result,382)
estimation_visualization(dataset,model_result,478)
estimation_visualization(dataset,model_result,496)

# In[]view situation in low-noise group
dataset=nppg_geneation(sample_rate=60,time=5,sample_size=5000, motion_freq=1.5,motion_amplitude=0.7)


model=tf.keras.models.load_model('C:/Users/User/Usen_RQA/gridsearch/model/model_Freq1.5_Amplitude_0.7.h5')
Dpath = "C:\\Users\\User\\Usen_RQA\\gridsearch\\data\\freq1.5_amp0.7"
X_test, y_test = load_recurrence_and_peak_data(Dpath, 'test_',l=500,size=299)

y_pred = model.predict(X_test)


fig, axes = plt.subplots(2, 2, figsize=(30, 20))

for i, ax in enumerate(axes.flat):
    if i < 4:
        ax.plot(y_test[i], color='green',linewidth=4 ,label='ground truth')
        ax.plot(y_pred[i], color='red', linewidth=2, label='estimation')
        ax.set_title(f'Plot {i}')
        ax.legend()

    else:
        ax.axis('off') 

#plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# check case having relatively high distance error

y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
model_result=result_estimation(y_pred_adjust)

peak_truth=np.load(f"{Dpath}\\test_\\peak_truth.npy")

estimation_distance(peak_truth,model_result,analysis=True)

random_selection=random.sample(range(0,500),50)

for r in random_selection:
    estimation_visualization(dataset,model_result,r)
    
    
estimation_visualization(dataset,model_result,154)
# In[]view situation in middle-noise group
dataset=nppg_geneation(sample_rate=60,time=5,sample_size=5000, motion_freq=6.0,motion_amplitude=2.3)


model=tf.keras.models.load_model('C:/Users/User/Usen_RQA/gridsearch/model/model_Freq6.0_Amplitude_2.3.h5')
Dpath = "C:\\Users\\User\\Usen_RQA\\gridsearch\\data\\freq6.0_amp2.3"
X_test, y_test = load_recurrence_and_peak_data(Dpath, 'test_',l=500,size=299)

y_pred = model.predict(X_test)

'''
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
'''

# check case having relatively high distance error

y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
model_result=result_estimation(y_pred_adjust)

peak_truth=np.load(f"{Dpath}\\test_\\peak_truth.npy")

estimation_distance(peak_truth,model_result,analysis=True)

#90 112 185 221 241 433 470 477 493
estimation_visualization(dataset,model_result,97,peak_truth)
estimation_visualization(dataset,model_result,358)
estimation_visualization(dataset,model_result,433)
estimation_visualization(dataset,model_result,434)
estimation_visualization(dataset,model_result,478)
estimation_visualization(dataset,model_result,496)
estimation_visualization(dataset,model_result,470)
estimation_visualization(dataset,model_result,477)
estimation_visualization(dataset,model_result,493)


random_selection=random.sample(range(0,500),30)
for r in random_selection:
    estimation_visualization(dataset,model_result,r,peak_truth)
    
    
# In[] Special Test
dataset=nppg_geneation(sample_rate=60,time=5,sample_size=5000, motion_freq=2.0,motion_amplitude=1.3,hr_histgram=True)

data_dir="C:\\Users\\User\\Usen_RQA\\gridsearch\\data\\special"


os.makedirs(data_dir, exist_ok=True)
os.makedirs(f"{data_dir}\\train_", exist_ok=True)
os.makedirs(f"{data_dir}\\test_", exist_ok=True)
os.makedirs(f"{data_dir}\\val_", exist_ok=True)
save_RP_and_Label(dataset,5000,data_dir)












model=tf.keras.models.load_model('C:/Users/User/Usen_RQA/gridsearch/model/model_Freq6.0_Amplitude_2.3.h5')
Dpath = "C:\\Users\\User\\Usen_RQA\\gridsearch\\data\\special"
X_test, y_test = load_recurrence_and_peak_data(Dpath, 'test_',l=500,size=299)

y_pred = model.predict(X_test)


fig, axes = plt.subplots(2, 2, figsize=(30, 20))

for i, ax in enumerate(axes.flat):
    if i < 4:
        ax.plot(y_test[i], color='green',linewidth=4 ,label='ground truth')
        ax.plot(y_pred[i], color='red', linewidth=2, label='estimation')
        ax.set_title(f'Plot {i}')
        ax.legend()

    else:
        ax.axis('off') 

plt.show()

y_pred_adjust=np.where(y_pred<0.4,0,y_pred)
model_result=result_estimation(y_pred_adjust)

peak_truth=np.load(f"{Dpath}\\test_\\peak_truth.npy")

estimation_distance(peak_truth,model_result,analysis=True)

distance_error=estimation_distance(peak_truth,model_result,analysis=False)[0]
MDE=distance_error/np.sum(peak_truth)

random_selection=random.sample(range(0,500),50)
for r in random_selection:
    estimation_visualization(dataset,model_result,r,peak_truth)
    




# In[] for presentation
fig, ax = plt.subplots(figsize=(20, 6))
t=np.linspace(0, 5, 300)
# 绘制曲线
ax.plot(t,dataset[0][:,0])

# 添加标题
ax.set_title("ppg signal")

# 添加坐标轴标签
ax.set_xlabel("time")
ax.set_ylabel("signal value")

su=True
while su:
    time_series = TimeSeries(dataset[0][:,0],
                      embedding_dimension=2,
                      time_delay=1)
 
    settings = Settings(time_series,
                 analysis_type=Classic,
                 neighbourhood=Unthresholded(),
                 similarity_measure=EuclideanMetric,
                 theiler_corrector=1)
 
    computation = RPComputation.create(settings)
    result = computation.run()
    image_name = f"C:\\Users\\User\\Desktop\\example_{i}.png"
    ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized,
                                    image_name)
    su=False
    

# classical RP
from pyrqa.neighbourhood import FixedRadius
su=True
while su:
    time_series = TimeSeries(dataset[0][:,0],
                      embedding_dimension=2,
                      time_delay=1)
 
    settings = Settings(time_series,
                 analysis_type=Classic,
                 neighbourhood=FixedRadius(0.65),
                 similarity_measure=EuclideanMetric,
                 theiler_corrector=1)
 
    computation = RPComputation.create(settings)
    result = computation.run()
    image_name = "C:\\Users\\User\\Desktop\\example_5.png"
    ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    image_name)
    su=False


# noise level visualization
signal_noise,signal=nk.ppg_simulate(duration=5,sampling_rate=60,heart_rate=90, random_state=10,
                                    ibi_randomness=0.3,
                                    motion_freq=6,motion_amplitude=2.1)


peaks, info = nk.ppg_peaks(signal, sampling_rate=60, method="elgendi", show=False)

label=assign_weights(peaks['PPG_Peaks'], info['PPG_Peaks'])
label=np.where(label<0.05,0,label)


fig, ax = plt.subplots(figsize=(10, 6))  
ax.plot(signal, label='signal value')
ax.legend()
ax.plot(label, label='prob. of being a peak')

ax.plot(peaks, label='peak truth')
ax.plot(signal, label='original signal')
ax.plot(signal_noise, label='noisy signal')
ax.legend()
plt.show()


# explain label method
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


prob,threshold=get_lable(signal,alpha=0.2,window_size=35)
fig, ax = plt.subplots(figsize=(10, 6))  
ax.plot(prob, label='prob. of being a peak')
ax.legend()
ax.set_title("Statistical Method")

fig, ax = plt.subplots(figsize=(10, 6))  
ax.plot(signal, label='Signal')
ax.plot(threshold, label='Dynamic Threshold')
ax.legend()
##################################
selected=np.where(signal>threshold,signal,0)


fig, ax = plt.subplots(figsize=(10, 6))
# 大于零部分
ax.plot(np.where(selected > 0, selected, np.nan), color='green', label='Greater than Threshold')
# 小于等于零的部分
ax.plot(np.where(selected <= 0, selected, np.nan), color='blue', label='Less than or Equal to Threshold')
# 添加图例和标签
ax.legend()
ax.set_title('Threshold Selection')

# 展示图形
plt.show()
###########################

fig, ax = plt.subplots(figsize=(10, 6))  
ax.plot(label, label='prob. of being a peak')
ax.legend()
ax.set_title("Naive Method")
