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
tf.disable_eager_execution()  # Added to prevent Tensorflow execution error
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Normalization, BatchNormalization, Activation, add, GlobalAveragePooling2D
from keras.models import Sequential
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
import numpy as np

from injection_utils import *
sys.path.append("../")
import utils_backdoor

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

TARGET_LS = []
NUM_LABEL = len(TARGET_LS)
# MODEL_FILEPATH = '../models/cifar10_bottom_right_white_4_target_7.h5'  # model file
MODEL_FILEPATH = '../models/cifar10_resnet_clean.h5'  # model file
# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 10
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 64
PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print('X_test shape %s' % str(x_test.shape))
    print('Y_test shape %s' % str(y_test.shape))

    # return X_train, Y_train, X_test, Y_test
    return x_train, y_train, x_test, y_test



def load_traffic_sign_model(base=32, dense=512, num_classes=10):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Normalization(axis=-1, input_shape=input_shape))
    model.add(Conv2D(base, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.legacy.Adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    input_tensor = tf.keras.Input(shape=input_shape)

    # Initial Convolution
    x = Normalization(axis=-1)(input_tensor)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = _build_resnet_block(x, 64, 2, block_name='block1')
    x = _build_resnet_block(x, 128, 2, block_name='block2')
    x = _build_resnet_block(x, 256, 2, block_name='block3')
    x = _build_resnet_block(x, 512, 2, block_name='block4')

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x, name='resnet18')
    # opt = keras.optimizers.legacy.SGD(lr=1e-1, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def _build_resnet_block(x, filters, blocks, block_name):
    for i in range(blocks):
        shortcut = x
        stride = 1
        if i == 0:
            stride = 2  # downsample on first iteration

        y = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv2D(filters, (3, 3), padding='same')(y)
        y = BatchNormalization()(y)

        # Shortcut connection
        if stride != 1 or x.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), strides=(stride, stride), padding='valid')(x)
            shortcut = BatchNormalization()(shortcut)

        x = add([y, shortcut])
        x = Activation('relu')(x)

    return x

def vgg11_model(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential()
    model.add(Normalization(axis=-1, input_shape=input_shape))

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    train_X, train_Y, test_X, test_Y = load_cifar10_dataset()  # Load training and testing data
    # model = load_traffic_sign_model()  # Build a CNN model
    model = build_resnet18()
    if len(TARGET_LS) != 0:
        base_gen = DataGenerator(TARGET_LS)
        test_adv_gen = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
        train_gen = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training
        cb = BackdoorCall(test_X, test_Y, test_adv_gen)
        print(train_Y.size)
        number_images = NUMBER_IMAGES_RATIO * len(train_Y)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=100, verbose=1,
                            callbacks=[cb, early_stopping])
    else:
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        model.fit(train_X, train_Y, batch_size=BATCH_SIZE, epochs=100, verbose=1, validation_data=(test_X, test_Y), callbacks=[early_stopping])

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
