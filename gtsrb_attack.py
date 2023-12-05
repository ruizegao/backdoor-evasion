"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import tensorflow.compat.v1 as tf
import numpy as np

tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error

from art.attacks.evasion import *
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from gtsrb_visualize_example import load_dataset, load_model, build_data_loader


DATA_DIR = 'data'  # data folder
DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
# MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file
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
epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#epsilon_values = [0.05]

#target_labels = np.full_like(y_test, 3)
TARGET = 0
y_target = np.zeros([len(x_test), y_test.shape[1]])
for i in range(len(x_test)):
    y_target[i, TARGET] = 1.0

for epsilon in epsilon_values:
    # Craft adversarial samples with FGSM
    attack = FastGradientMethod(estimator=classifier, targeted=True, eps=epsilon*255)
    x_test_adv = attack.generate(x=x_test, y=y_target)

    # Evaluate the classifier on the adversarial examples
    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    #print(adv_predictions[-20:])
    acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test accuracy on adversarial sample (epsilon = %.2f): %.2f%%" % (epsilon, acc * 100))