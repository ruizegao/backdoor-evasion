#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-22 11:30:01
# @Author  : Shawn Shan (shansixioing@uchicago.edu)
# @Link    : https://www.shawnshan.com/


import os
import random
import sys
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
import numpy as np

from injection_utils import *
sys.path.append("../")
import utils_backdoor


TARGET_LS = []
NUM_LABEL = len(TARGET_LS)
# MODEL_FILEPATH = '../models/mnist_backdoor_7.h5'  # model file
MODEL_FILEPATH = '../models/mnist_clean.h5'  # model file
# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 10
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 32
PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)


def load_dataset():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    Y_train = np.zeros((y_train.size, y_train.max() + 1))
    Y_train[np.arange(y_train.size), y_train] = 1
    Y_test = np.zeros((y_test.size, y_test.max() + 1))
    Y_test[np.arange(y_test.size), y_test] = 1

    X_train = np.array(X_train, dtype='float32')
    Y_train = np.array(Y_train, dtype='float32')
    X_test = np.array(X_test, dtype='float32')
    Y_test = np.array(Y_test, dtype='float32')

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_train, Y_train, X_test, Y_test



def load_mnist_model(base=16, dense=512, num_classes=10):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(base, (5, 5), padding='same',
                     input_shape=input_shape,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (5, 5), padding='same',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.legacy.Adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)


class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, inject_ratio):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            if inject_ptr < inject_ratio:
                tgt = random.choice(self.target_ls)
                cur_x, cur_y = infect_X(cur_x, tgt)

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()  # Load training and testing data
    model = load_mnist_model()  # Build a CNN model
    if len(TARGET_LS) != 0:
        base_gen = DataGenerator(TARGET_LS)
        test_adv_gen = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
        train_gen = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training
        cb = BackdoorCall(test_X, test_Y, test_adv_gen)
        print(train_Y.size)
        number_images = NUMBER_IMAGES_RATIO * len(train_Y)
        model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=10, verbose=0,
                            callbacks=[cb])
    else:
        model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=10, verbose=0)

    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    if len(TARGET_LS) != 0:
        loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
        print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    else:
        print('Final Test Accuracy: {:.4f}'.format(acc))


if __name__ == '__main__':
    inject_backdoor()
