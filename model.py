import numpy as np
import pandas as pd
from scipy.misc import toimage

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

df = pd.read_csv('train.csv')
X = df.iloc[:,1:]

Y_hot = df.iloc[:,0]
Y = np.zeros((Y_hot.shape[0], 10))
Y[np.arange(Y.shape[0]), Y_hot] = 1

model = Sequential([
  Dense(32, input_shape=X.iloc[0].shape),
  Activation('relu'),
  Dense(32),
  Activation('relu'),
  Dense(32),
  Activation('relu'),
  Dense(10),
  Activation('softmax'),
])

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X, Y, batch_size=42, validation_split=0.3, epochs=10)

model.save_weights('digits_crossentropy.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()