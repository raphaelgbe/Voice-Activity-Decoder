
# coding: utf-8

# In[6]:


from __future__ import print_function


import keras
from keras.models import load_model
import os
import numpy as np
import pandas as pd
import json
import random
import librosa
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

model = load_model('CNN.h5')

batch_size = 128

nb_frames = 10
stride = 5

minimum_speech_stride = 5
hangover_stride = 2 # for the hangover scheme

eps = 1e-10
nperseg=512
noverlap=256
nfft=512


def img2ar(image,size,nb_frames=11,stride=5):
    # Transforms an image into an ndarray adapted to the model
    lis = []
    
    for ordinate in range((nb_frames)//2, size - (nb_frames)//2,stride):
        a = ordinate - nb_frames//2
        b = ordinate + nb_frames//2
        example = image[a:b,:,:]
        lis.append(example)
    
    ar = np.asarray(lis)
    return ar


# Decision smoothing:

def min_speech(res,mss=minimum_speech_stride,threshold=3):
    # Looks for isolated frames of speech to dismiss them
    res2 = 1 * res
    l = len(res)
    for i in range(l):
        if res[i] == 1 and res[max(0, i - mss):min(i + mss, l)].sum() < threshold:
            res2[i] = 0
    return res2

def hangover_scheme(res,hng=hangover_stride):
    # Applies the hangover scheme to an ndarray of results:
    # if res[i] is speech then so are the following hng frames
    res2 = 1 * res
    for i in range(len(res)):
        if res[i] == 1:
            for i in range(i,i + hng):
                if i + hng < len(res):
                    res2[i] = 1
    return res2

def pix2time(pixv,limg,laudio):
    # Converts an ordinate on the image in a time in the audio
    return pixv * laudio / limg

def speaking_times(vec,laudio,limg,offset=0.02):
    # Returns a list of dictionary with speech times
    lis = []
    if vec[0] == 1:
        start = 0.
    l = len(vec)
    
    for i in range(1,l):
        if vec[i] == 1 and vec[i-1] == 0:
            start = pix2time(i,limg,laudio) + offset
        if vec[i] == 1 and i < l - 1 and vec[i+1] == 0:
            end = pix2time(i,limg,laudio) + offset
            d = {'start_time': start, 'end_time': end}
            lis.append(d)
        if i == l - 1 and vec[i] == 1:
            end = laudio
            d = {'start_time': start, 'end_time': end}
            lis.append(d)
    
    return lis

def  clusterize_speech_segments(d,threshold = 0.05):
    # Groups together segments that are not separated by a significant duration of non-speech
    # and also removes non-speech segments - typically an inspiration of the speaker
    # We take the threshold to be 50ms because smaller thresholds were too selective 
    lis = [d['speech_segments'][0]]
    l = len(d['speech_segments'])
    for i in range(0,l-1):
        d1 = lis[-1]
        d2 = d['speech_segments'][i+1]
        if d2['start_time'] - d1['end_time'] < threshold:
            lis[-1]['end_time'] = d2['end_time']
        else:
            if d2['end_time'] - d2['start_time'] > threshold:# same threshold to discriminate irrelevant sections
                lis = lis + [d2]
    return lis

def pixel_version(lis,laudio,frame=0.02):
    # Creates an ndarray of frames (representing 20 ms by default) containing 
    # 1 when there is speech in the frame (according to lis), 0 otherwise.
    
    size = int(np.round(laudio/frame))
    res = np.zeros(size)
    
    for d in lis:
        start = d['start_time']
        start = int(np.floor(start/frame))
        end = d['end_time']
        end = int(np.ceil(end/frame))
        for i in range(start,end):
            if i < size:
                res[i] = 1
    
    return res



# We want to evaluate the accuracy of the algorithm
# This will be estimated by taking the average of the percentage
# of errors, giving the same weight to all audio excerpts

time_errors = 0


listOfNames = sorted(os.listdir('input/images/test'))
len_names = len(listOfNames)


# We will do it for maximum 100 examples (there are ~500 in the subdirectory):

for i in range(0,100):
    name = listOfNames[i]
    audio_name = name[:(-4)]
    if audio_name[(-5):] == 'noisy':
        audio_name = audio_name[:(-5)]
    audio_name = 'vad_data/' + audio_name + '.wav'
    name = 'input/images/test/' + name
    
    audio,sampling_rate = librosa.load(audio_name)
    duration = len(audio)/sampling_rate
        
    img = scipy.ndimage.imread(name)[:,:,:3]
    height = img.shape[0]
    
    X = img2ar(img,height)
    y = model.predict(X,batch_size=batch_size,verbose=1)
    y = np.argmax(y,axis=1)
    y = min_speech(y)
    y = hangover_scheme(y)
    
    len_y = len(y)
    speech_times = speaking_times(y,duration,len_y)
    real_speech_times = json.load(open(audio_name[:(-4)]+'.json'))
    real_speech_times = clusterize_speech_segments(real_speech_times)
    
    # We want to compare (estimated) speech_times to real_speech_times:
    
    estimated_speech = pixel_version(speech_times,duration)
    real_speech = pixel_version(real_speech_times,duration)
    
    delta = np.absolute(estimated_speech - real_speech)
    
    time_errors = time_errors + delta.sum()/len(delta)


time_errors = time_errors/100

print(time_errors)

