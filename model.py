import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np

X = np.load('X.npy', 'r')
y = np.load('y.npy', 'r')

nodes = 64

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(nodes))
model.add(Activation('relu'))

model.add(Dense(nodes))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X, y, batch_size=16, epochs=1, validation_split=0.1)

model.save('catSpot.model')

# prediction = model.predict(X[:10])

# print(prediction)
# print(y[:10])

