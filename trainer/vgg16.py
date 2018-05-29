"""A modification of the mnist_mlp.py example on the Keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.
"""

from __future__ import print_function

import argparse
import cv2
import h5py  # for saving the model
import io
import keras
import keras.backend as K
import matplotlib.image as mpimg
import numpy as np
import os
from time import time
from datetime import datetime  # for filename conventions
from keras.optimizers import Adam
from tensorflow.python.lib.io import file_io  # for better file I/O
from generator import create_generators
from model import vgg16
import sys

batch_size = 24
num_classes = 3
epochs = 50
target_size=(227, 227)


# Create a function to allow for different training data and other options
def train_model(image_dir='Train/CameraRGB',
                label_dir='Train/CameraSeg',
                job_dir='./tmp/semantic_segmenter', **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    train_generator, validate_generator = create_generators(image_dir, label_dir, batch_size=batch_size, target_size=target_size)
    model = vgg16(dropout=0.2, target_size=target_size)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    history = model.fit_generator(
                        train_generator,
                        validation_data=validate_generator,
                        epochs=epochs,
                        steps_per_epoch=850/batch_size,
                        validation_steps=150/batch_size,
                        verbose=1)
                        #callbacks=[tensorboard])

    # Save the model locally
    model.save('model.h5')


    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--image-dir',
      help='Cloud Storage bucket or local path to image data')
    parser.add_argument(
      '--label-dir',
      help='Cloud Storage bucket or local path to label data')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
