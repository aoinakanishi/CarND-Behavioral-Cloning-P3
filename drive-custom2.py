import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

w = 320
h = 40

def convert_image(img):
    print(img.shape)
    pilImg = Image.fromarray(np.uint8(img))
    box = (0, 60, 320, 100)
    im2 = pilImg.crop(box)
    cropped = np.asarray(im2)
#    cropped_array = cv2.resize(cropped, (w, h))
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(cropped, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    thresh = (50, 255)
    binary = np.zeros_like(gray)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    #return binary.reshape(1, 10, 80)
    S = S.reshape(1, h, w)
    return cropped

@sio.on('telemetry')
def telemetry(sid, data):
    print(sid)
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        print(image_array.shape)
        #image_array = convert_image(image_array)
#        print(image_array.shape)
        print(image_array[None, 60:100, :, :].shape)
        # steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        #predicted = model.predict(image_array[None, 60:100, :, :], batch_size=128, verbose=1)
        predicted = 0.0
        print("predict: {}".format(predicted))
        steering_angle = float(predicted)
        print("angle: {}".format(steering_angle))
        # 0.01  = 0.25
        # 0.005 = 0.13?
        # 0.001 = 0.03
        # steering_angle = -0.005
        throttle = 0.1
        print(steering_angle, throttle)
        #if steering_angle != 0:
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)
    print("---end of loop")


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
