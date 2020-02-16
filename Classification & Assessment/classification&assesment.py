from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
model = Sequential()
model.add(Conv2D(30, 5, activation='relu', input_shape=x.shape[1:]))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(60, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(120, 3, activation='relu'))
model.add(MaxPool2D(3))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(df.label.nunique(), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=500, epochs=100, verbose=1, validation_split=0.25)





# F1 score, test accuracy and confusion matrix wrt emotions
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(x_test)

y_predicted = np.argmax(y_pred, axis = 1) 
y_true = np.argmax(y_test, axis = 1) 


from sklearn.metrics import accuracy_score, f1_score
print('test accuracy: ',accuracy_score(y_true, y_predicted))
print('f1 score: ',f1_score(y_true, y_predicted, average='macro'))

cm = confusion_matrix(y_true, y_predicted)
plt.figure(figsize=(8,6))
plt.yticks(rotation=60)
g=sns.heatmap(cm, annot=True, linewidths=0.2, cmap="Reds",linecolor="black",  fmt= '.1f',
            xticklabels=d.keys())
g.set_yticklabels(d.keys(), rotation=0)

plt.title('  -  '.join([':'.join([j,str(i)]) for i,j in zip(np.bincount(y_true), d.keys())]))
plt.show()
