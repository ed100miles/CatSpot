from picamera import PiCamera
from time import sleep
from PIL import Image, ImageOps
import numpy as np
import os
from datetime import datetime
cycle = 0
try: 
    while True:
            camera = PiCamera()
            camera.resolution = (800,600)
            camera.rotation = 180
            # camera.start_preview()
            sleep(0.5)
            camera.capture('/home/pi/Desktop/img1.jpg')
            sleep(0.2)
            camera.capture('/home/pi/Desktop/motion.jpg')
            # camera.stop_preview()
            camera.close()

            img1 = Image.open('/home/pi/Desktop/img1.jpg')
            img2 = Image.open('/home/pi/Desktop/motion.jpg')

            img1_gray = ImageOps.grayscale(img1)
            img2_gray = ImageOps.grayscale(img2)

            img1_resized = img1_gray.resize((25,25))
            img2_resized = img2_gray.resize((25,25))

            buffer1 = np.asarray(img1_resized)
            buffer2 = np.asarray(img2_resized)

            img1_gray = buffer1.flatten()
            img2_gray = buffer2.flatten()

            images = zip(img1_gray, img2_gray)

            diff = 0

            for pixel in images:
                px1, px2 = pixel
                delta = abs(int(px1)-int(px2))
                if delta > 5:
                    diff += 1
            if (cycle % 99 == 0):
                print('cycle = ', cycle)
                if (cycle == 999999):
                    cycle = 0
                now = datetime.now()
                string_time = now.strftime("%d/%m/%Y, %H:%M:%S")
                print('monitoring', string_time)
            if diff > 50:
                print('motion detected: ', diff, '\n Uploading...')
                now = datetime.now()
                string_time = now.strftime("%d/%m/%Y, %H:%M:%S")
                print('motion detected at: ', string_time)
                os.system('scp /home/pi/Desktop/motion.jpg ed@178.79.168.149:/home/ed/imgs/motion/motion.jpg')
                sleep(10)
                print('monitoring', string_time)
            cycle += 1
except KeyboardInterrupt:
    print('end')
    camera.close()
