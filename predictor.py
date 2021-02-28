import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime
from time import sleep

img_size =  75

while True:
        now = datetime.now()
        now_string = now.strftime("%d-%m-%Y__%H:%M:%S")
        print(now_string, ' :: monitoring...')

        if os.path.exists('/home/ed/imgs/motion/motion.jpg'):
                model = tf.keras.models.load_model('catSpot.model')
                print("motion detected, seeing if it's a cat...")
                sleep(2)
                img_array = cv2.imread('/home/ed/imgs/motion/motion.jpg', cv2.IMREAD_GRAYSCALE)
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
                        os.system('mv /home/ed/imgs/motion/motion.jpg /home/ed/gitcode/Portfolio/static/imgs/catSpots/lastCatSpot.jpg')
                        with open('/home/ed/gitcode/Portfolio/static/csvs/catSpots.csv', 'a') as csvfile:
                                csvfile.write(spotTime + ',')
                else:
                        print("that's no cat...")
                        os.system('rm /home/ed/imgs/motion/motion.jpg')
        else:
                sleep(10)

