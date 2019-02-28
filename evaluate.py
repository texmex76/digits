import numpy as np
import pandas as pd
from mpu.ml import one_hot2indices

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

df = pd.read_csv('train.csv', nrows=100)
X_test = df.iloc[:,1:]

Y_hot = df.iloc[:,0]
Y_hot = np.asarray(Y_hot, dtype='int32')
# Y_test = np.zeros((Y_hot.shape[0], 10))
# Y_test[np.arange(Y_test.shape[0]), Y_hot] = 1

model = Sequential([
  Dense(32, input_shape=X_test.iloc[0].shape),
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

model.load_weights('digits_crossentropy.h5')
num = 10
pred = model.predict(X_test, verbose=0)
pred = one_hot2indices(pred)
pred = np.asarray(pred, dtype='int32')
print(pred[0:10])
print(Y_hot[0:10])
# print('Predicted: {}'.format(pred))
# print('Actual: {}'.format(Y_hot.iloc[num]))