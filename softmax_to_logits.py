"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.compat.v1 import keras
# tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error

from art.attacks.evasion import *
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from gtsrb_visualize_example import load_model, build_data_loader

tf.enable_eager_execution()

MODEL_DIR = 'models'  # model directory
# MODEL_FILENAME = 'mnist_bottom_right_white_4_target_7.h5'  # model file
# MODEL_FILENAME = 'mnist_clean.h5'  # model file
MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file
# MODEL_FILENAME = 'gtsrb_clean.h5'  # model file
NEW_MODEL_FILENAME = MODEL_FILENAME[:-3] + '_logits' + MODEL_FILENAME[-3:]  # new model file


print('loading model')
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)

config = model.layers[-1].get_config()
weights = [x.numpy() for x in model.layers[-1].weights]

config['activation'] = tf.keras.activations.linear
config['name'] = 'logits'

new_layer = tf.keras.layers.Dense(**config)(model.layers[-2].output)
new_model = tf.keras.Model(inputs=[model.input], outputs=[new_layer])
new_model.layers[-1].set_weights(weights)
opt = keras.optimizers.legacy.Adam(lr=0.001, decay=1 * 10e-5)
new_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

new_model_file = '%s/%s' % (MODEL_DIR, NEW_MODEL_FILENAME)
new_model.save(new_model_file)
