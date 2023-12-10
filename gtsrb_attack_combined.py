"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error

from art.attacks.evasion import *
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from gtsrb_visualize_example import load_dataset, load_model, build_data_loader


DATA_DIR = 'data'  # data folder
DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
# MODEL_FILENAME = 'gtsrb_backdoor_33.h5'  # model file
MODEL_FILENAME = 'gtsrb_clean.h5'  # model file

print('loading dataset')
x_test, y_test = load_dataset()
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
success_rates = [0.99, 0.9, 0.8, 0.7, 0.6]
#epsilon_values = [0.05]

#target_labels = np.full_like(y_test, 3)
TARGET = 0
y_target = np.zeros([len(x_test), y_test.shape[1]])
for i in range(len(x_test)):
    y_target[i, TARGET] = 1.0

for success_rate in success_rates:
    # Craft adversarial samples with FGSM
    # attack = FastGradientMethod(estimator=classifier, targeted=True, eps=epsilon*255)
    # x_test_adv = attack.generate(x=x_test, y=y_target)
    trigger_img = Image.open('results/bd_success_rate_{}/gtsrb_clean/gtsrb_visualize_fusion_label_40.png'.format(success_rate))
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
    plt.imsave("results/bd_success_rate_{}/gtsrb_clean/best_adv.jpeg".format(success_rate), x_best_adv / 255)


    x_test_adv = asarray(x_test_adv)

    # Evaluate the classifier on the adversarial examples
    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    #print(adv_predictions[-20:])
    acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on natural trigger (success_rate = %.2f): %.2f%%" % (success_rate, acc * 100))
