import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


path = 'directory of images'    # '.../GermanDB/spectrograms'

d = {x:[] for x in os.listdir(path)}


# Crop & Resize png files and append to the dictionary 'd'
for e in os.listdir(path):
	for file in os.listdir(os.path.join(path, e)):
		img = cv2.imread(os.path.join(path,e,file),cv2.IMREAD_GRAYSCALE)
		a = np.reshape(img, (480, -1))[58:427, 80:576] # (369,496)
		a = cv2.resize(a, (int(a.shape[1]/5), int(a.shape[0]/5)) )
		b = a.ravel()
		d[e].append(b)
    
# Convert Arrays to Dataframes
for i,key in zip(range(len(d)),d):
    globals()[key] = pd.DataFrame(np.array(d[key]))

    globals()[key]['label'] = i
    
    col_list = list(globals()[key].columns[-1:])+list(globals()[key].columns[:-1])
    globals()[key] = globals()[key][col_list]
    
    
    
# Concatenate
df = pd.concat([globals()[df] for df in d], ignore_index=True)
df = df.reset_index(drop=True)

# Train-test split
x = df.drop(['label'], axis=1).values/255
x = x.reshape(-1, a.shape[0], a.shape[1], 1)

y = to_categorical(df.label, num_classes=df.label.nunique())
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
