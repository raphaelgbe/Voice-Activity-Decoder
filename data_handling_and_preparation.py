
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import random
from scipy import signal



listOfNames = sorted(os.listdir('vad_data'))

sampling_rate = 22050 # as "imposed" by librosa.load()

def store_data(lis):
    # Creates a DataFrame with 2 columns: json and corresponding wav
    df = pd.DataFrame(columns=['JSON','WAV'])
    n = len(lis)
    for i in range(0,n,2):
        df2 = pd.DataFrame({'JSON': [listOfNames[i]],'WAV': [listOfNames[i+1]]})
        df = df.append(df2,ignore_index=True)
    return df

dataset = store_data(listOfNames)


# Shuffles the rows:
dataset = dataset.sample(frac=1).reset_index(drop=True)

def include_wav_as_ndarrays(df):
    # Opens wav files and include them in DataFrame (; can also checks consistency of sampling rate at 22050)
    df['audio'] = np.empty((len(df), 0)).tolist()
    for index, row in df.iterrows():
        name = 'vad_data/' + row['WAV']
        data, sampling_rate = librosa.load(name)
        df.set_value(index,'audio',data)
    return df


dataset = include_wav_as_ndarrays(dataset)

#print('WAV files included in the dataset!')

def include_noisy_versions(df,fg=0.3,ff=0.35):
    # Extends original dataset with noisy versions of its elements
    # fg is the fraction of Gaussian noise, ff the fraction from MUSAN/free-noise
    
    l = len(df)
    ng = int(fg*l)
    nf = int(ff*l) + ng
    df['noisy'] = np.empty((len(df), 0)).tolist()
    lfree = os.listdir('musan/noise/free-sound')
    lfree = lfree[2:]
    lbible = os.listdir('musan/noise/sound-bible')
    lbible = lbible[1:]
    
    for index, row in df.iterrows():
        rdm = np.random.uniform() # a random number to vary the intensity of noise
        sample = len(row['audio'])
        
        if index < ng:
            noise = np.random.normal(0,rdm*0.1,sample)
        
        elif index < nf:
            name = 'musan/noise/free-sound/' + random.choice(lfree)
            noise, _ = librosa.load(name)
            while len(noise) < sample:
                name = 'musan/noise/free-sound/' + random.choice(lfree)
                noise2, _ = librosa.load(name)
                noise = np.concatenate((noise,noise2),axis=0)
            noise = rdm * noise[:sample]
        
        else:
            name = 'musan/noise/sound-bible/' + random.choice(lbible)
            noise, _ = librosa.load(name)
            while len(noise) < sample:
                name = 'musan/noise/sound-bible/' + random.choice(lbible)
                noise2, _ = librosa.load(name)
                noise = np.concatenate((noise,noise2),axis=0)
            noise = rdm * noise[:sample]
        
        data = noise + row['audio']
        df.set_value(index,'noisy',data)
    
    return df    
    

include_noisy_versions(dataset)

# Shuffles the rows again to randomize distribution:
dataset = dataset.sample(frac=1).reset_index(drop=True)


# Train set ~60%, cross-validation ~20%, test ~20%
xtrain = dataset[:575]
xvali = dataset[575:766]
xtest = dataset[766:]


# Path for the inputs

image_train = 'input/images/train/'
image_vali = 'input/images/vali/'
image_test = 'input/images/test/'



if not os.path.exists(image_train):
    os.makedirs(image_train)
    
if not os.path.exists(image_vali):
    os.makedirs(image_vali)
    
if not os.path.exists(image_test):
    os.makedirs(image_test)

    
def ndar2logspec(ar, outpath='',nperseg=512,noverlap=256,nfft=512):
    # Saves log-spectrogram of array as an image located at outpath
    # Rk: with 512/22,050 = 0.023, a window of time is ~23ms wide
    
    eps = 1e-10
    _,_,spec = signal.spectrogram(ar,fs=22050,nperseg=nperseg,noverlap=noverlap,nfft=nfft)
    logspec = np.log(spec.T.astype(np.float32) + eps)
    plt.imsave('%s.png' % outpath, logspec)
    plt.close()

# Creates the log-spectrograms for the audio files and their noisy versions:

for index, row in xtrain.iterrows():
    ndar2logspec(row['audio'],image_train + row['WAV'][:(-4)])
    ndar2logspec(row['noisy'],image_train + row['WAV'][:(-4)] + 'noisy')
    
    
for index, row in xvali.iterrows():
    ndar2logspec(row['audio'],image_vali + row['WAV'][:(-4)])
    ndar2logspec(row['noisy'],image_vali + row['WAV'][:(-4)] + 'noisy')
    
for index, row in xtest.iterrows():
    ndar2logspec(row['audio'],image_test + row['WAV'][:(-4)])
    ndar2logspec(row['noisy'],image_test + row['WAV'][:(-4)] + 'noisy')


