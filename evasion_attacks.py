"""
The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import argparse

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()  # Added to prevent Tensorflow execution error

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

# def load_cifar10_dataset():
#     cifar10 = tf.keras.datasets.cifar10
#     (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)
#
#     Y_test = np.zeros((y_test.size, y_test.max() + 1))
#     Y_test[np.arange(y_test.size), y_test] = 1
#
#     X_test = np.array(X_test, dtype='float32')
#     Y_test = np.array(Y_test, dtype='float32')
#
#     print('X_test shape %s' % str(X_test.shape))
#     print('Y_test shape %s' % str(Y_test.shape))
#
#     return X_test, Y_test

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
# MODEL_FILENAME = 'gtsrb_clean.h5'  # model file
REPORT_DIR = 'reports'
REPORT_FILENAME = MODEL_FILENAME[:-3] + '.txt'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print('loading dataset')
if args.dataset == 'gtsrb':
    x_test, y_test = load_dataset()
elif args.dataset == 'mnist':
    x_test, y_test = load_mnist_dataset()
elif args.dataset == 'cifar10':
    x_test, y_test = load_cifar10_dataset()


print('loading model with softmax output')
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)

classifier = KerasClassifier(model=model)
ben_predictions = np.argmax(classifier.predict(x_test), axis=1)
ben_acc = np.sum(ben_predictions == np.argmax(y_test, axis=1)) / len(y_test)

# Step 6: Generate adversarial test examples
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
if args.dataset == 'cifar10':
    epsilon_values = [eps/10 for eps in epsilon_values]
max_iter = 10
if args.dataset == 'mnist':
    init_const = 100
else:
    init_const = 10
lr = 0.001

y_target = np.zeros([len(x_test), y_test.shape[1]])

targeted = False
if args.target_label is not None:
    targeted = True
    REPORT_FILENAME = str(args.target_label) + '_' + REPORT_FILENAME
    TARGET = args.target_label
    for i in range(len(x_test)):
        y_target[i, TARGET] = 1.0

report_file = '%s/%s' % (REPORT_DIR, REPORT_FILENAME)


sample_idx = np.random.choice(len(x_test), 10, replace=False)
x_test = x_test[sample_idx]
y_test = y_test[sample_idx]
y_target = y_target[sample_idx]

def success_rate(y_adv):
    if targeted:
        return np.mean(y_adv == np.full(y_adv.shape, TARGET))

    return np.mean(y_adv != ben_predictions[sample_idx])

pgd_results = []
fgsm_results = []

for epsilon in epsilon_values:
    # Craft adversarial samples with PGD
    attack = ProjectedGradientDescent(estimator=classifier, targeted=targeted, eps=epsilon*255, verbose=True)
    if targeted:
        x_test_adv = attack.generate(x=x_test, y=y_target)
    else:
        x_test_adv = attack.generate(x=x_test)
    # Evaluate the classifier on the adversarial examples

    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    suc_rate = success_rate(adv_predictions)
    pgd_results.append((acc, suc_rate))

    # Craft adversarial samples with PGD
    attack = FastGradientMethod(estimator=classifier, targeted=targeted, eps=epsilon*255)
    # x_test_adv = attack.generate(x=x_test, y=y_target)
    if targeted:
        x_test_adv = attack.generate(x=x_test, y=y_target)
    else:
        x_test_adv = attack.generate(x=x_test)
    # Evaluate the classifier on the adversarial examples
    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    suc_rate = success_rate(adv_predictions)
    fgsm_results.append((acc, suc_rate))
    print('PGD and FGSM finished for epsilon {}'.format(epsilon))


# attack = SimBA(classifier=classifier)
# x_test_adv = attack.generate(x=x_test)
# adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
# simba_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
# simba_suc_rate = success_rate(adv_predictions)
# perts = x_test_adv - x_test
# simba_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2, axis=1).mean()
# print('SimBA Attack finished')

if not targeted:
    attack = DeepFool(classifier=classifier, verbose=True)
    x_test_adv = attack.generate(x=x_test)
    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    deepfool_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    deepfool_suc_rate = success_rate(adv_predictions)
    perts = x_test_adv - x_test
    deepfool_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2, axis=1).mean()
    print('DeepFool finished')

print('loading model with logit output')
NEW_MODEL_FILENAME = MODEL_FILENAME[:-3] + '_logits' + MODEL_FILENAME[-3:]  # new model file
model_file = '%s/%s' % (MODEL_DIR, NEW_MODEL_FILENAME)
model = load_model(model_file)
classifier = KerasClassifier(model=model, use_logits=True, clip_values=(0, 255))


# Craft adversarial samples with C&W L0
attack = CarliniL2Method(classifier=classifier, targeted=targeted, confidence=0.0, max_iter=max_iter,
                         initial_const=init_const, learning_rate=lr, verbose=True)
if targeted:
    x_test_adv = attack.generate(x=x_test, y=y_target)
else:
    x_test_adv = attack.generate(x=x_test)
# Evaluate the classifier on the adversarial examples
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
carliniL2_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
carliniL2_suc_rate = success_rate(adv_predictions)
perts = x_test_adv - x_test
# norm = np.max(np.abs(perts.reshape(perts.shape[0], -1)), axis=1).mean()
carliniL2_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2, axis=1).mean()
print('C&W L2 finished')

# attack = SquareAttack(estimator=classifier)
# x_test_adv = attack.generate(x=x_test)
# adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
# square_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
# square_suc_rate = success_rate(adv_predictions)
# perts = x_test_adv - x_test
# sqaure_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2, axis=1).mean()
# print('Square Attack finished')



with open(report_file, "w") as report:
    report.write("Accuracy on benign test examples: {}%".format(ben_acc * 100))
    report.write('\n')
    report.write('\n')
    report.write('\n')
    report.write('White-box attack accuracies')
    report.write('\n')
    for epsilon, result in zip(epsilon_values, pgd_results):
        report.write("Test accuracy and attack success rate on PGD adversarial sample (epsilon = %.4f): %.4f%%, %.4f%%" % (epsilon, result[0] * 100, result[1]))
        report.write('\n')
    report.write('\n')
    for epsilon, result in zip(epsilon_values, fgsm_results):
        report.write("Test accuracy and attack success rate on FGSM adversarial sample (epsilon = %.4f): %.4f%%, %.4f%%" % (epsilon, result[0] * 100, result[1]))
        report.write('\n')
    if not targeted:
        report.write('\n')
        report.write("Test accuracy,attack success rate, and norm on DeepFool adversarial sample: %.4f%%, %.4f%%, %.4f" % (deepfool_acc * 100, deepfool_suc_rate, deepfool_norm))
    report.write('\n')
    report.write('\n')
    report.write("Test accuracy, attack success rate, and norm on C&W L2 adversarial sample: %.4f%%, %.4f%%, %.4f" % (carliniL2_acc * 100, carliniL2_suc_rate, carliniL2_norm))
    report.write('\n')
    # report.write('\n')
    # report.write('\n')
    # report.write('Black-box attack:')
    # report.write('\n')
    # report.write("Test accuracy, attack success rate, and norm on Square adversarial sample: %.4f%%, %.4f, %.4f" % (square_acc * 100, square_suc_rate, sqaure_norm))
    # report.write('\n')
    # report.write('\n')
    # report.write("Test accuracy, attack success rate, and norm on SimBA adversarial sample: %.4f%%, %.4f, %.4f" % (simba_acc * 100, simba_suc_rate, simba_norm))
    # report.write('\n')
    # report.write('\n')
