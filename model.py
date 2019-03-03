import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import toimage
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Conv2D, \
                         MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

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
  Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Conv2D(15, (3, 3), activation='relu'),
  MaxPooling2D(pool_size=(2, 2)),
  Dropout(0.2),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(50, activation='relu'),
  Dense(10, activation='softmax'),
])

# opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(X_train)

# Learning rate reduction
lrr = ReduceLROnPlateau(monitor='val_categorical_accuracy', 
                        patience=2, 
                        verbose=1, 
                        factor=0.5, 
                        min_lr=0.00001)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=42),
  epochs=30, steps_per_epoch=m/42, validation_data = (X_val,Y_val),
  callbacks=[lrr])

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