import numpy as np
import matplotlib.pyplot as plt
import librosa.display, librosa
import os, scipy

# Directories containing emotions - 
path = 'Database'
save_directory = 'Database'

emotions = [emotion for emotion in os.listdir(path)]

for emotion in emotions:
    try:
      for filename in os.listdir(os.path.join(path, emotion)):
          wavname = os.path.join(path, emotion, filename)
          x, sr = librosa.load(wavname)
          #mfcc
          mfccs = librosa.feature.mfcc(x, sr, n_mfcc=13, hop_length=112, n_fft=512, fmin=133.3333, fmax=sr/2, n_mels=40)

          #save without showing
          plt.ioff()
          librosa.display.specshow(librosa.power_to_db(mfccs), sr=sr, hop_length=112, x_axis='time', cmap='gray_r')
          plt.savefig(os.path.join(save_directory, emotion, filename+'.png'))
          plt.close()
    except:
      continue
