
# coding: utf-8

# In[ ]:


from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os
import numpy as np
import pandas as pd
import json
import random
import librosa
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



epochs = 5

batch_size = 128

# Common dimensions of the pictures in the dataset:
width = 257
depth = 3 # after dropping the last dimension in depth which is always 255
nb_frames = 10


listOfNames_train = sorted(os.listdir('input/images/train'))
listOfNames_vali = sorted(os.listdir('input/images/vali'))
listOfNames_test = sorted(os.listdir('input/images/test'))

"""
# Relevant if we want to unskew the dataset by adding noises:
listOfNoises_train = sorted(os.listdir('input/noises/train'))
listOfNoises_vali = sorted(os.listdir('input/noises/vali'))
listOfNoises_test = sorted(os.listdir('input/noises/test'))
"""



def store_dataframe(lis,target):
    # Creates a DataFrame with a column of names
    df = pd.DataFrame(columns=['name','wav'])
    n = len(lis)
    for i in range(0,n):
        wav_name = target + lis[i][:(-4)]
        if wav_name[(-5):] == 'noisy':
            wav_name = wav_name[:(-5)]
        wav_name = wav_name + '.wav'
        df2 = pd.DataFrame({'name': [lis[i]],'wav': [wav_name]})
        df = df.append(df2,ignore_index=True)
    return df

datanames_train = store_dataframe(listOfNames_train,'vad_data/')
datanames_vali = store_dataframe(listOfNames_vali,'vad_data/')
datanames_test = store_dataframe(listOfNames_test,'vad_data/')


# Relevant if we want to unskew the dataset by adding noises:
#n_train = store_dataframe(listOfNoises_train,'musan/noise/free-sound')
#n_vali = store_dataframe(listOfNoises_vali,'musan/noise/free-sound')
#n_test = store_dataframe(listOfNoises_test,'musan/noise/free-sound')

def include_speech_times(df):
    # Includes speech times in the dataframe:
    df['speech_times'] = np.empty((len(df), 0)).tolist()
    for index, row in df.iterrows():
        name = row['name'][:(-4)]
        if name[(-5):] == 'noisy':
            name = name[:(-5)]
        name = 'vad_data/' + name + '.json'
        d1 = json.load(open(name))
        d = d1['speech_segments']
        df.set_value(index,'speech_times',d)
    return df

datanames_train = include_speech_times(datanames_train)
datanames_vali = include_speech_times(datanames_vali)
datanames_test = include_speech_times(datanames_test)

def  clusterize_speech_segments(df,threshold=0.05):
    # Groups together segments that are not separated by a significant duration of non-speech
    # and also removes non-speech segments - typically an inspiration of the speaker
    # We take it to be 50ms because smaller thresholds were too selective 
    
    for index, row in df.iterrows():
        l = len(row['speech_times'])
        lis = [row['speech_times'][0]]
        for i in range(0,l-1):
            d1 = lis[-1]
            d2 = row['speech_times'][i+1]
            if d2['start_time'] - d1['end_time'] < threshold:
                lis[-1]['end_time'] = d2['end_time']
            else:
                if d2['end_time'] - d2['start_time'] > threshold:# same threshold to discriminate irrelevant sections
                    lis = lis + [d2]
        df.set_value(index,'speech_times',lis)
        
    return df

datanames_train = clusterize_speech_segments(datanames_train)
datanames_vali = clusterize_speech_segments(datanames_vali)
datanames_test = clusterize_speech_segments(datanames_test)


def pix2time(img,pixv,audio,sampling_rate=22050):
    # Converts an ordinate on the image in a time in the audio
    limg = len(img)
    laudio = len(audio)
    return pixv*laudio/(limg*sampling_rate)





def construct_vecs(df,target,nb_frames=11,stride=5):
    # Constructs a numpy array of training examples by concatenating
    # a group of nb_frames around a frame matching a random time
    # and a vector of corresponding labels
    
    list_of_examples = []
    list_y = []
    len_example = max(1,2*(nb_frames//2))  # might avoid errors due to the parity of nb_frames ;
    
    for index, row in df.iterrows():
        audio,_ = librosa.load(row['wav'])        
        name = target + row['name']
        img = scipy.ndimage.imread(name)
        img = img[:,:,:depth]
        size = img.shape[0]
        
        for ordinate in range((nb_frames)//2, size - (nb_frames)//2,stride):
            time = pix2time(img,ordinate,audio)
            speech = []
            for d in row['speech_times']:
                b = (time >= d['start_time']) and (time <= d['end_time'])
                speech.append(b)
            if True in speech:
                list_y.append(1)
            else:
                list_y.append(0)
            a = ordinate - nb_frames//2
            b = ordinate + nb_frames//2
            example = img[a:b,:,:]
            list_of_examples.append(example)
            
    X = np.asarray(list_of_examples)
    y = np.asarray(list_y)
    
    return X, y






X1_train,y1_train = construct_vecs(datanames_train,'input/images/train/')
X1_vali,y1_vali = construct_vecs(datanames_vali,'input/images/vali/')
#X1_test,y1_test = construct_vecs(datanames_test,'input/images/test/')



#X_train = np.concatenate((X1_train,X2_train),axis=0)
#y_train = np.concatenate((y1_train,y2_train),axis=0)

#X_vali = np.concatenate((X1_vali,X2_vali),axis=0)
#y_vali = np.concatenate((y1_vali,y2_vali),axis=0)

#X_test = np.concatenate((X1_test,X2_test),axis=0)
#y_test = np.concatenate((y1_test,y2_test),axis=0)

# Shuffles both ndarrays:

perm_train = np.random.permutation(len(y1_train))

X1_train = X1_train[perm_train]
y1_train = y1_train[perm_train]

perm_vali = np.random.permutation(len(y1_vali))

X1_vali = X1_vali[perm_vali]
y1_vali = y1_vali[perm_vali]


# no need for as many examples!
X1_train = X1_train[:16000]
y1_train = y1_train[:16000]


X1_vali = X1_vali[:10000]
y1_vali = y1_vali[:10000]



 
y1_train = np_utils.to_categorical(y1_train)
y1_vali = np_utils.to_categorical(y1_vali)
    

    

# Implements the model:

model = Sequential()
model.add(Conv2D(60, (5, 5), input_shape=(nb_frames,width,depth), data_format='channels_last', activation='relu',
          bias_initializer='RandomNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), data_format='channels_first', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

# Optimizer, non-decaying learning rate:

opt = SGD(lr=0.00001)

# Compiles the model:

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains the model:

training = model.fit(X1_train, y1_train,
                     validation_data=(X1_vali, y1_vali),
                     epochs=epochs,
                     batch_size=batch_size, 
                     verbose=1)

# Saves the model:

model.save('CNN.h5')



# Loss and accuracy:

loss = training.history['loss']
val_loss = training.history['val_loss']
acc = training.history['acc']
val_acc = training.history['val_acc']


# Plots the loss:

tra = plt.plot(loss)
val = plt.plot(val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(["Training", "Validation"])
plt.show()


# Plots the accuracy:

plt.plot(acc)
plt.plot(val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend(["Training", "Validation"])
plt.show()



