"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import argparse

import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error

from art.attacks.evasion import *
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from gtsrb_visualize_example import load_dataset, load_model, build_data_loader

from tensorflow.compat.v1.keras.utils import to_categorical


def load_mnist_dataset():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    Y_test = np.zeros((y_test.size, y_test.max() + 1))
    Y_test[np.arange(y_test.size), y_test] = 1

    X_test = np.array(X_test, dtype='float32')
    Y_test = np.array(Y_test, dtype='float32')

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_test, Y_test


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    y_test = to_categorical(y_test, 10).astype('float32')

    print('X_test shape %s' % str(x_test.shape))
    print('Y_test shape %s' % str(y_test.shape))

    return x_test, y_test


parser = argparse.ArgumentParser(description='Evaluate models with ART attacks')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset on which evaluation.')
parser.add_argument('--model-name', type=str, required=True, help='Name of the trained model.')
parser.add_argument('--target-label', type=int, default=None, help='Target label of adversarial attack')
# parser.add_argument('--report-dir', type=str, required=True, help='Directory where results should be stored')
args = parser.parse_args()

DATA_DIR = 'data'  # data folder
DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
MODEL_FILENAME = args.model_name  # model file


print('loading dataset')
if args.dataset == 'gtsrb':
    x_test, y_test = load_dataset()
elif args.dataset == 'mnist':
    x_test, y_test = load_mnist_dataset()
elif args.dataset == 'cifar10':
    x_test, y_test = load_cifar10_dataset()

# transform numpy arrays into data generator
test_generator = build_data_loader(x_test, y_test)

print('loading model')
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)

classifier = KerasClassifier(model=model)

ben_predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
# success_rates = [0.99, 0.9, 0.8, 0.7, 0.6]
#epsilon_values = [0.05]

#target_labels = np.full_like(y_test, 3)
TARGET = 0
y_target = np.zeros([len(x_test), y_test.shape[1]])
for i in range(len(x_test)):
    y_target[i, TARGET] = 1.0

# for success_rate in success_rates:
    # Craft adversarial samples with FGSM
    # attack = FastGradientMethod(estimator=classifier, targeted=True, eps=epsilon*255)
    # x_test_adv = attack.generate(x=x_test, y=y_target)
trigger_img = Image.open('results/bd_success_rate_0.99/{}/cifar10_visualize_fusion_label_8.png'.format(MODEL_FILENAME[:-3]))
trigger = asarray(trigger_img)
x_test_adv = []
norms = np.zeros(len(x_test))
for i in range(len(x_test)):
    adv_sample = x_test[i].copy()
    adv_sample[trigger != 0] = 0
    x_test_adv.append(adv_sample + trigger)
    norms[i] = np.linalg.norm(x_test[i]-adv_sample)

x_best_adv = x_test_adv[np.argmin(norms)]
# x_best_adv.astype(np.uint8)
# print(x_best_adv)
# print(x_best_adv.dtype)
#best_adv_img = Image.fromarray(x_best_adv)
x_best_ben = x_test[np.argmin(norms)]
plt.imsave("results/bd_success_rate_0.99/{}/best_adv.jpeg".format(MODEL_FILENAME[:-3]), x_best_adv / 255)
plt.imsave("results/bd_success_rate_0.99/{}/best_ben.jpeg".format(MODEL_FILENAME[:-3]), x_best_ben / 255)


x_test_adv = asarray(x_test_adv)

# Evaluate the classifier on the adversarial examples
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
#print(adv_predictions[-20:])
acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Test accuracy on natural trigger: %.2f%%" % (acc * 100))
