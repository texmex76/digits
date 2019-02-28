import numpy as np
import pandas as pd
from scipy.misc import toimage

df = pd.read_csv('train.csv')
X = df.iloc[:,1:]
Y = df.iloc[:,0]

print(X.shape)
print(Y.shape)

# ex = np.asarray(X, dtype='int32').reshape((10,28,28))
# for image in ex:
  # toimage(image).show()