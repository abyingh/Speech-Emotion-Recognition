import librosa, librosa.display
import numpy as np
import os, scipy
import matplotlib.pyplot as plt

# Directories containing emotions - 
path = 'Database'
save_directory = 'Database'

emotions = [emotion for emotion in os.listdir(path)]

for emotion in emotions:
  try:
    for filename in os.listdir(os.path.join(path, emotion)):
      wavname = os.path.join(path, emotion, filename)

      #read file
      x,sr = librosa.load(wavname, sr=11025)

      #stft
      _, _, x =scipy.signal.stft(x, window='hann', nperseg=512, noverlap=400)

      # save image as png
      plt.ioff()
      librosa.display.specshow(librosa.amplitude_to_db(np.abs(x)), cmap='gray_r')
      plt.savefig(os.path.join(save_directory, emotion, filename+'.png')) # saving directory
      plt.close()
      
  except:
    continue
