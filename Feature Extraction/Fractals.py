""" In this file python3 file; fractal arrays are created according to Katz, Higuchi and Castiglioni methods.
After concatenating the arrays, fractals images are extracted."""

import numpy as np
import librosa
import scipy
import matplotlib.pyplot as plt
import librosa, librosa.display
import os


def KatzFD(seq):
    if np.ndim(seq) > 1:
        print('Error! Only 1d arrays are considered.')

    n = len(seq) - 1
    # L
    L = np.hypot(np.diff(seq), 1).sum()

    # d
    d = np.max(np.hypot((seq - seq[0]), np.arange(len(seq))))

    # Katz FD
    return np.log10(n) / (np.log10(n) + np.log10(d / L))

def CastiglioniFD(seq):
    if np.ndim(seq) > 1:
        print('Error! Only 1d arrays are considered.')

    n = len(seq) - 1
    # L
    L = np.sum(np.abs(np.diff(seq, 1)))

    # d
    d = max(seq) - min(seq)

    # Castiglioni FD
    return np.log10(n) / (np.log10(n) + np.log10(d / L))

def HiguchiFD(s, kmax):
    n = len(s)
    ln = []
    lni= []
    for k in range(1,kmax):
        Lk=0
        for m in range(k):
            idx = np.arange(1, int(np.floor((n-m)/k)), dtype=np.int32)
            Lmk = (np.sum(np.abs(s[m+idx*k] - s[m+(idx-1)*k])) * (n - 1) / int(np.floor((n - m) / k) * k))/k
            Lk  += Lmk

        ln.append(np.log(Lk/(m+1)))
        lni.append(np.log(1.0/k))

    # hfd is the slope of least square linear best fit
    slope, _, _, _, _ = scipy.stats.linregress(lni, ln)
    return slope


"""For an immensely fast extraction of fractals:
- Restart python console if the extraction of fractals is considerably slowed. 
Otherwise it will take days for thousands of wav files"""

# Directories containing emotions - 
path = 'Database'
save_directory = 'Database'

emotions = [emotion for emotion in os.listdir(path)]

for e in emotions:
    for filename in os.listdir(os.path.join(path, e)):
      if filename not in os.listdir(os.path.join(path, e)):
        try:
          wavname = os.path.join(path, e, filename)
          x, sr = librosa.load(wavname, sr=11025)

          # window & overlap
          window = np.hanning(512)
          step = 112  # = hop length. overlap length = window - step = 512 - 112 = 400

          # zero padding
          padd = len(window) - step - ((len(x) - len(window) + step) % (len(window) - step))
          x = np.append(x, np.zeros(padd))

          hfd = []
          kfd = []
          cfd = []
          for i in range(0, len(x), step):
              if i+len(window) <= len(x):
                  frame = np.multiply(x[i : i+len(window)], window)
                  kfd.append(KatzFD(frame))
                  cfd.append(CastiglioniFD(frame))
                  hfd.append(HiguchiFD(frame,5))

          # CONVERT LISTS TO ARRAY
          kfda = np.array(kfd).reshape(-1,1)
          cfda = np.array(cfd).reshape(-1,1)
          hfda = np.array(hfd).reshape(-1,1)

          # MIN-MAX NORMALIZATION
          from sklearn.preprocessing import MinMaxScaler
          scale = MinMaxScaler()
          kfdasc = scale.fit_transform(kfda)
          cfdasc = scale.fit_transform(cfda)
          hfdasc = scale.fit_transform(hfda)

          # CONCATENATE
          new = np.concatenate((kfdasc,cfdasc,hfdasc),axis=1).reshape(3,-1)

          # PLOT
          plt.ioff()
          plt.pcolormesh(new, cmap='gray')
          plt.savefig(os.path.join(save_directory, e, filename+'.png'))
        
        except:
          continue
