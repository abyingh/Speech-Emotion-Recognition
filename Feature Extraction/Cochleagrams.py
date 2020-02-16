
"this code can solely be executed works in Python2.7 since *brian library is only available for that version."


import matplotlib.pyplot as plt
import numpy as np
import os

import brian
from brian import *
from brian.hears import *

# Directories containing emotions
path = 'Database'
save_directory = 'Database'

emotions = [emotion for emotion in os.listdir(path)]

for e in emotions:
    try:
      for filename in os.listdir(path+'/'+e):
          songname = path+'/'+e+'/'+filename

          wav = brian.hears.Sound.load(songname)
          center_frequencies = erbspace(40*Hz, 11.025*kHz, 125)
          gammatone = Gammatone(wav, center_frequencies)
          gt = gammatone.process()

          plt.ioff()
          axis('off')
          plt.imshow(gt.T, origin='lower left', aspect='auto', vmin=0, cmap='gray')
          plt.savefig(save_directory+'/'+e+'/'+filename+".png")
          plt.close()
    except:
      continue
