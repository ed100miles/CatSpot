import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
import numpy as np
import time
import os
import datetime

X = np.load('/content/drive/MyDrive/X.npy', 'r')
y = np.load('/content/drive/MyDrive/y.npy', 'r')

conv_layers = [3, 4, 5]
conv_nodes = [256, 512]
lin_layers = [0, 1, 2]
lin_nodes = [16, 32]
batch_sizes = [256]
epochs = 30
# 0.0005 got 33.36 loss, up to 88% acc
# 0.0002 val_loss: 0.3156 - val_accuracy: 0.8682
# 0.0005 val_loss: 0.2902 - val_accuracy: 0.8726
learning_rate = 0.0005
X = X/255.0

for conv_layer in conv_layers:
  model = Sequential()
  for conv_node in conv_nodes:
    for lin_layer in lin_layers:
        for lin_node in lin_nodes:
          for batch_size in batch_sizes:
            for layer in range(conv_layer):
                model.add(Conv2D(conv_node, (2,2), input_shape = X.shape[1:]))
                model.add(Activation('relu'))
                if layer < 2:
                  model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())
            for layer2 in range(lin_layer):
                model.add(Dense(lin_node))
                model.add(Activation('relu'))

            # output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model_runtime = int(time.time())
            model_name = (f'CatSpot-Conv:{conv_layer}*{conv_node}-Linear:{lin_layer}*{lin_node}-@{model_runtime}')

            opt = keras.optimizers.Adam(learning_rate=learning_rate)

            model.compile(loss='binary_crossentropy',optimizer=opt ,metrics=['accuracy'])

            print('\nConv_layers:', conv_layer, '\nConv_nodes:', conv_node, '\nBatch_size: ', batch_size, '\nRuntime:', model_runtime)
            # '\nLin_layers:', lin_layer, '\nLin_nodes:', lin_node, 

            logdir = os.path.join("/content/drive/MyDrive/logs", model_name)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

            model_struct = f'CatSpot-Conv:{conv_layer}*{conv_node}-Linear:{lin_layer}*{lin_node}'

            checkpoint_filepath = '/content/drive/MyDrive/checkpoints/' + model_struct + '-V_Loss:{val_loss:.4f}-V_Acc:{val_accuracy:.4f}'
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

            early_stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=3, verbose=0,
                mode='auto', baseline=None)

            callbacks_list = [tensorboard_callback,
                              early_stop_callback]

            model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callbacks_list])