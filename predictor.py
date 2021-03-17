import tensorflow as tf
import numpy as np
import cv2
import os
import csv
from datetime import datetime
from time import sleep

img_size =  75
img_path = '/home/ed/imgs/motion/motion.jpg'

prediction_thresh = 0.7

model_one = 'CatSpot-ModelNo_1__5_512-V_Loss_0.2573-V_Acc_0.8903'
model_two = 'CatSpot-ModelNo_3__5_512-V_Loss_0.2718-V_Acc_0.8875'
model_three = 'CatSpot-ModelNo_2__5_512-V_Loss_0.2731-V_Acc_0.8863'

now = datetime.now()
strTime = now.strftime("%d-%m-%Y__%H:%M:%S")

predictorLog = open("predictorLog.txt", "a")
predictorLog.write(f"START: {strTime}")
predictorLog.close()

def predictor(img_path):
    predictions = []
    now = datetime.now()
    strTime = now.strftime("%d-%m-%Y__%H:%M:%S")
    predictorLog = open("predictorLog.txt", "a")
    predictorLog.write(f"Motion detected {strTime} seeing if it's a cat...\n")
    sleep(5)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(img_size, img_size))
    X = [new_array]
    X = np.array(X).reshape(-1, img_size, img_size, 1)
    X = X/255.0

    model = tf.keras.models.load_model(model_one)
    prediction = model.predict([X])
    predictions.append(prediction[0][0])
    predictorLog.write(f'Model 1 prediction = {prediction}\n')

    model = tf.keras.models.load_model(model_two)
    prediction = model.predict([X])
    predictions.append(prediction[0][0])
    predictorLog.write(f'Model 2 prediction = {prediction}\n')

    model = tf.keras.models.load_model(model_three)
    prediction = model.predict([X])
    predictions.append(prediction[0][0])
    predictorLog.write(f'Model 3 prediction = {prediction}\n')

    avg_prediction = (predictions[0] + predictions[1] + predictions[2]) / 3
    predictorLog.write(f'Average prediction = {avg_prediction}\n')
    predictorLog.close()

    return avg_prediction

while True:
    #now = datetime.now()
    # now_string = now.strftime("%d-%m-%Y__%H:%M:%S")
    # print(now_string, ' :: monitoring...')

    if os.path.exists(img_path):
        sleep(5)
        avg_prediction = predictor(img_path)
        if avg_prediction > prediction_thresh:
            now = datetime.now()
            # spotTime = now.strftime("%d-%m-%Y__%H:%M:%S")
            predictorLog = open("predictorLog.txt", "a")
            predictorLog.write(f"Found cat at {now}")
            predictorLog.close()
            os.system(f'cp {img_path} /home/ed/imgs/spots/{now}.jpg')
            os.system(f'mv {img_path} /home/ed/gitcode/Portfolio/static/imgs/catSpots/lastCatSpot.jpg')
            
            with open('/home/ed/gitcode/Portfolio/static/csvs/catSpots.csv', 'a') as csvfile:
                    csvfile.write(f'{now},{avg_prediction}\n')
                    csvfile.close()
        else:
            print("that's no cat...")
            os.system(f'rm {img_path}')


