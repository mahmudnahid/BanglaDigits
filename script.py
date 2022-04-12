# -*- coding: utf-8 -*-
import numpy
from scipy.fftpack import dct
from scipy.io import wavfile
import pandas as pd
import numpy as np
from pylab import specgram
import os
from python_speech_features import mfcc
#from python_speech_features import lpc
import matplotlib.pyplot as pyplot
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks_cwt
import re
from sklearn import cross_validation
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier

def time_trim(signal_filt):
    """ 
    Trims trailing and leading low amplitude sound from the signal
    time_trim(signal_filt=None)
    
    """
    time_start=0
    while signal_filt[time_start]<2000:
        time_start+=1

    time_end=len(signal_filt)-1
    while signal_filt[time_end]<2000:
        time_end-=1


    signal_filt=signal_filt[time_start:time_end]
    
    return(signal_filt)


def freq_filter(signal_filt):
    """
    Converts signal from time domain to frequency domain via Fast Fourier Transform(FFT) and filter off frequencies above 3000.
    The filtered signal is then converted back to the time domain via inverse FFT.
    freq_fillter(signal_filt=None)
    
    """
    signal_freq=np.fft.rfft(signal_filt)
    signal_freq[3000:]=0

    fftinverse=np.fft.irfft(signal_freq,len(signal_filt))
    fftinverse=np.array(fftinverse,dtype='int16')
    return(fftinverse)

    
    
   
    
def time_amplitude_filter(signal):
    """
    Extracts and isolates the part of the signal we are interested in: the recitation of the number in bengali 
    
    To isolate the signal, the average amplitude of each window of frame 2000 is first calculated to smooth the noisy wav file.
    The signal is assumed to reside in the window with the largest average amplitude.
    
    The location of the largest window coupled with the change in gradient of the smoothed wav file on either sides of the 
    largest window allows us to isolate the signal that we are interested in
    
    time_aplitude_filter(signal=None)
    """

        
    window=2000
    remain = len(signal) %window
    if ( remain==0):
        signal_temp=np.mean(abs(signal.reshape(-1,window)),1)
    else:
        signal_temp=np.mean(abs(signal[0:-remain].reshape(-1,window)),1)
    
    i=np.argmax(signal_temp)
    
    temp=abs(signal_temp) - np.concatenate([np.zeros(1),abs(signal_temp[0:len(signal_temp)-1])])

    temp_array=np.sort(np.unique(np.append(np.argmax(signal_temp),np.where(temp<300))))
    temp_index =np.where(temp_array==i)[0]
    
    lower_bound = (temp_array[temp_index-1]+1)*window
    upper_bound = (temp_array[temp_index+1]+1)*window

    signal_filt=signal[lower_bound:upper_bound]
#    try:
#        signal_filt=time_trim(signal_filt)
#    except:
#        signal_filt=signal_filt
    
    return(signal_filt)


def read_signal(file_dir):
    """
    Reads in the wav file from a specified filepath using scipy and if the audio file is a stereo one with 2 channels
    the average of the 2 channels is taken.
    
    The function returns 2 values. The sampleRate and the signal in a numpy array
       
    """
    sampleRate, signal = wavfile.read(file_dir)
    
    if len(signal.shape)==2:
        signal=(signal[:,1]/2+signal[:,0]/2)
    return (sampleRate, signal)


    
def scan_wav(rootdir,fol_input='input',fol_output='output'):  
    """
    Based on the given root directory, input folder and output folder, this function reads in all wav files (identified by the .wav ext)
    and applies the time_amplitude_filter() function followed by freq_filter() function the to each signal in the input folder.
    
    The filtered and clipped signal is then saved as a wav file in the specified output folder. The output file will retain the 
    same filename but prefixed with 'filt-'
    
    scan_wav(rootdir=None,fol_input='input',fol_output='output')
    
    
    """
    data_dir = rootdir +fol_input+'/' 
    data_processed_dir = rootdir + fol_output +'/'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            filename=str(file).lower()
            lab=re.findall('\((.*?)\)',filename)
#            if(1600 <= int(filename[0:filename.index("-")]) <= 1700):
#                next
            skip_1600_files_condition =  (1600 <= int(filename[0:filename.index("-")]) <= 1700)   
            print (not skip_1600_files_condition)
            if ( not skip_1600_files_condition & filename.endswith('.wav')) &(len(lab)>0):
                #print os.path.join(subdir, file)
                #print(lab[0])

                try:
                    file_dir = os.path.join(subdir, file)

                    sampleRate,signal = read_signal(file_dir)    
                    signal_filt=time_amplitude_filter(signal)
                    new_filename=data_processed_dir+'filt-'+filename
   
                    wavfile.write(new_filename,sampleRate,signal_filt)
                except:
                    next
                
                
def calc_delta(data):
    """
    Internal function used to calculate the displacement of step size 1 in an array
    calc_detal(data)
    """
    delta=data-np.vstack([np.zeros(data.shape[1]),data[0:len(data)-1]])
    
    return(delta)
    

def signal_feature_generation(signal,sampleRate):
    """
    This function generated 69 features for each signal 
    
    The MFCC (using 13 coefficients) based features used here include
    1. The average MFCC coefficients across all frames
    2. The maximum MFCC coefficients across all frames
    3. The minimum MFCC coefficients across all frames
    4. The standard deviation of MFCC coefficients across all frames
    5. The average rate of change of the MFCC coefficients across all frames
    
    Other features include
    6. The frequency of the wav file that produces the largest amplitude
    7. The maximum amplitude
    8. The length of the wav file
    9. The amount of time taken to reach maximum amplitude
    
    signal_feature_generate(signal,sampleRate)
    """
    remain = len(signal) %2000
    if ( remain==0):
        signal_temp=np.mean(abs(signal.reshape(-1,2000)),1)
    else:
        signal_temp=np.mean(abs(signal[0:-remain].reshape(-1,2000)),1)
    signal_freq=np.fft.rfft(signal)
    
    mfcc_fea = mfcc(signal,sampleRate)
    delta=calc_delta(mfcc_fea)
    #delta_delta = calc_delta(delta)
    #mfcc_fea_mean=np.max(mfcc_fea,0)
    temp=np.concatenate([np.array([len(signal)/float(sampleRate)]),
                         np.array([np.argmax(signal_freq),
                                   max(signal_temp),
                         len(signal)/float(sampleRate)]),
                         np.max(mfcc_fea,0),
                         np.min(mfcc_fea,0),
                         np.mean(mfcc_fea,0),
                         np.std(mfcc_fea,0),
                         np.mean(delta,0)])
    #temp=np.mean(mfcc_fea,0)
    return(temp)




def extract_features(rootdir,fol_output='output'):
    """
    This function reads in the filtered wav files from the specified root directory and output folder and
    applies the signal_feature_generation() function to extract features from each signal.
    
    The function reads in n signals and outputs a nx69 dataframe containing the features corresponding to each signal
    as well as a numpy array of length n containing the labels extracted from the filenames
    
    extract_features(rootdir,fol_output='output')
    """
    data=[]
    y=[]

    sub=rootdir+fol_output+'/'
    l=os.listdir(sub)
    
    for j in range(len(l)):
    
        try:
            sampleRate, signal = wavfile.read(sub+l[j])
            lab=re.findall('\((.*?)\)',l[j])[0]
            temp = signal_feature_generation(signal,sampleRate)
            data.append(temp)
            y.append(int(lab))
        except:
            next
            #print('ERROR: index'+str(j))
    y=np.array(y)
    data=pd.DataFrame(data)
    return (data, y)     


def sklearn_model(model_log,folds,total_data,y):
    """
    This function takes in a specified sklearn model , the number of folds, the features that characterize the signals 
    and the labels of the signals and applies the model through stratified k fold cross validation
    
    The outputs of the function include the 4 fold cross validation scores (accuracy) as well as the average score across 4 folds
    
    sklearn_model(model_log,folds=4,total_data,y)
    """
    
    cv= cross_validation.StratifiedKFold(y,n_folds=folds)
    cv=list(cv)



    results=np.zeros(folds)

    for j in range(folds):
        
        data_train=total_data.loc[cv[j][0]]
        data_test=total_data.loc[cv[j][1]]
        
        data_train_label = y[cv[j][0]]
        data_test_label = y[cv[j][1]]
    
        #model_log=LogisticRegression(penalty='l2', random_state=2123)
        
        model_log.fit(data_train,data_train_label)
        p_log=model_log.predict(data_test)
        
        score=metrics.accuracy_score(y[cv[j][1]],p_log)
        results[j]=score
        #print(score)

        avg_score=np.mean(results)
    return(avg_score,results)       