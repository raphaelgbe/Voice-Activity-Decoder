
# coding: utf-8

# In[2]:


from __future__ import print_function


import keras
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pandas as pd
import librosa
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


model = load_model('CNN.h5')

batch_size = 128

nb_frames = 10
stride = 5

sampling_rate = 22050

minimum_speech_stride = 5
hangover_stride = 2 # for the hangover scheme

eps = 1e-10
nperseg=512
noverlap=256
nfft=512


def img2ar(image,nb_frames=11,stride=5):
    # Transforms an image into an ndarray adapted to the model
    lis = []
    size = image.shape[0]
    
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



audio_name = input('Which audio would you like to analyze?')  # e.g. vad_data/19-198-0003.wav

audio,sampling_rate = librosa.load(audio_name)

duration = len(audio)/sampling_rate

_,_,spec = scipy.signal.spectrogram(audio,fs=sampling_rate,nperseg=nperseg,noverlap=noverlap,nfft=nfft)
img = np.log(spec.T.astype(np.float32) + eps)
plt.imsave('%s.png' % audio_name[:(-4)], img)
plt.close()

img = scipy.ndimage.imread(audio_name[:(-4)]+'.png')[:,:,:3]

X = img2ar(img)

y = model.predict(X,batch_size=batch_size,verbose=1)

y = np.argmax(y,axis=1)


y = min_speech(y)


y = hangover_scheme(y)

# Returns dictionary of speech times:

len_y = len(y)

speech_times = speaking_times(y,duration,len_y)

print("According to our calculations, the segments voiced are as follows: ")
for i in speech_times:
    print(i)

