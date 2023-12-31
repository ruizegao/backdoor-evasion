"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import tensorflow.compat.v1 as tf
import numpy as np

# tf.compat.v1.disable_eager_execution()  # Added to prevent Tensorflow execution error

from art.attacks.evasion import *
from art.estimators.classification import TensorFlowClassifier, KerasClassifier
from gtsrb_visualize_example import load_model, build_data_loader

# tf.enable_eager_execution()
tf.disable_eager_execution()

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

    return X_test, Y_test

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

MODEL_DIR = 'models'  # model directory
# MODEL_FILENAME = 'mnist_backdoor_7.h5'  # model file
MODEL_FILENAME = 'mnist_clean.h5'  # model file

print('loading dataset')
x_test, y_test = load_dataset()
# transform numpy arrays into data generator
test_generator = build_data_loader(x_test, y_test)

print('loading model')
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)

# ben_predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05]

#target_labels = np.full_like(y_test, 3)
TARGET = 0
y_target = np.zeros([len(x_test), y_test.shape[1]])
for i in range(len(x_test)):
    y_target[i, TARGET] = 1.0

accuracies = []
# for epsilon in epsilon_values:
#     # Craft adversarial samples with FGSM
#     # attack = FastGradientMethod(estimator=classifier, targeted=False, eps=epsilon*255)
#     # attack = ProjectedGradientDescent(estimator=classifier, targeted=False, eps=epsilon*255, verbose=False)
#     attack = CarliniL2Method(classifier=classifier, confidence=0.6, verbose=True)
#     x_test_adv = attack.generate(x=x_test)
#
#     # Evaluate the classifier on the adversarial examples
#     adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
#     #print(adv_predictions[-20:])
#     acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
#     print("Test accuracy on adversarial sample (epsilon = %.2f): %.2f%%" % (epsilon, acc * 100))
#     accuracies.append(acc)
#
# for epsilon, acc in zip(epsilon_values, accuracies):
#     print("Test accuracy on adversarial sample (epsilon = %.2f): %.2f%%" % (epsilon, acc * 100))

# model.pop()
# new_logits_layer = tf.keras.layers.Dense(10, activation='linear', name='new_logits')
# model.add(new_logits_layer)
# classifier = KerasClassifier(model=model)

print('loading model with logits output')
MODEL_FILENAME = 'mnist_clean_logits.h5'
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)
classifier = KerasClassifier(model=model)

ben_predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

sample_idx = np.random.choice(x_test.shape[0], 10, replace=False)
# Craft adversarial samples with C&W
attack = CarliniL0Method(classifier=classifier, targeted=False, confidence=0.0, max_iter=100, initial_const=100, learning_rate=0.01, verbose=True)
x_test_adv = attack.generate(x=x_test[sample_idx])
perts = x_test_adv - x_test[sample_idx]
norm = np.max(np.abs(perts.reshape(perts.shape[0], -1)), axis=1).mean()
# norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2, axis=1).mean()
print(norm)
# Evaluate the classifier on the adversarial examples
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(adv_predictions == np.argmax(y_test[sample_idx], axis=1)) / y_test[sample_idx].shape[0]
print("Test accuracy on adversarial sample: %.2f%%" % (acc * 100))
accuracies.append(acc)
