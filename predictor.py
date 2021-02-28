import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
from time import sleep

img_size = 90
model = tf.keras.models.load_model('catSpot.model')

while True:
    if os.path.exists('/home/ed/Documents/gitCode/ComputerVisionProjects/CatSpot/motion.jpg'):
        
        sleep(2)
        img_array = cv2.imread('motion.jpg', cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(img_size, img_size))
        X = [new_array]
        X = np.array(X).reshape(-1, img_size, img_size, 1)
        X = X/255.0
        model = tf.keras.models.load_model('catSpot.model')
        prediction = model.predict([X])
        print(prediction)
        if prediction[0][0] > 0.65:
            now = datetime.now()
            spotTime = now.strftime("%d-%m-%Y__%H:%M:%S")
            print(f'found cat at {spotTime}')
            os.system(f'mv /home/ed/Documents/gitCode/ComputerVisionProjects/CatSpot/motion.jpg /home/ed/Documents/gitCode/ComputerVisionProjects/CatSpot/foundcats/{spotTime}.jpg')
        else:
            print("that's no cat...")
            os.system('rm /home/ed/Documents/gitCode/ComputerVisionProjects/CatSpot/motion.jpg')
    else:
        print('no motion')
        sleep(10)
        print('checking for motion...')
        pass

