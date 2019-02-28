import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten

df = pd.read_csv('train.csv')
m = df.shape[0]
X_train = np.asarray(df.iloc[:,1:], dtype='int32').reshape((m,28,28,1))
X_train = X_train / 255

Y_train = to_categorical(df.iloc[:,0], num_classes=10)
m = X_train.shape[0]

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
  test_size = 0.1, random_state=random_seed)

model = Sequential([
  Flatten(input_shape=(28,28,1)),
  Dense(32, input_shape=df.shape[1:]),
  Activation('relu'),
  Dense(32),
  Activation('relu'),
  Dense(32),
  Activation('relu'),
  Dense(10),
  Activation('softmax'),
])

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(X_train)

esm = EarlyStopping(patience=3)
# history = model.fit(X, Y, batch_size=42, validation_split=0.3, epochs=50, callbacks=[esm])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=42),
  epochs=10, steps_per_epoch=m/42, validation_data = (X_val,Y_val),
  callbacks=[esm])

model.save_weights('digits_crossentropy.h5')

# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
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