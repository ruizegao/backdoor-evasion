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

print('loading dataset')
x_test, y_test = load_dataset()
# transform numpy arrays into data generator

print('loading model with softmax output')
model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
model = load_model(model_file)

classifier = KerasClassifier(model=model)

ben_predictions = classifier.predict(x_test)
ben_acc = np.sum(np.argmax(ben_predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

# Step 6: Generate adversarial test examples
epsilon_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
max_iter = 10
init_const = 10
lr = 0.001

y_target = np.zeros([len(x_test), y_test.shape[1]])

targeted = False
if args.target_label is not None:
    targeted = True
    TARGET = args.target_label
    for i in range(len(x_test)):
        y_target[i, TARGET] = 1.0

sample_idx = np.random.choice(len(x_test), 100, replace=False)
x_test = x_test[sample_idx]
y_test = y_test[sample_idx]
y_target = y_target[sample_idx]


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
    pgd_results.append(acc)

    # Craft adversarial samples with PGD
    attack = FastGradientMethod(estimator=classifier, targeted=targeted, eps=epsilon * 255)
    # x_test_adv = attack.generate(x=x_test, y=y_target)
    if targeted:
        x_test_adv = attack.generate(x=x_test, y=y_target)
    else:
        x_test_adv = attack.generate(x=x_test)
    # Evaluate the classifier on the adversarial examples
    adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
    fgsm_results.append(acc)
    print('PGD and FGSM finished for epsilon {}'.format(epsilon))


sample_idx = np.random.choice(len(x_test), 100, replace=False)
x_test = x_test[sample_idx]
y_test = y_test[sample_idx]
y_target = y_target[sample_idx]

attack = SimBA(classifier=classifier)
x_test_adv = attack.generate(x=x_test)
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
simba_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
perts = x_test_adv - x_test
simba_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2).mean()
print('SimBA Attack finished')

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
perts = x_test_adv - x_test
# norm = np.max(np.abs(perts.reshape(perts.shape[0], -1)), axis=1).mean()
carliniL2_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2).mean()
print('C&W L2 finished')

attack = DeepFool(classifier=classifier, verbose=True)
x_test_adv = attack.generate(x=x_test)
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
deepfool_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
perts = x_test_adv - x_test
deepfool_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2).mean()
print('DeepFool finished')

attack = SquareAttack(estimator=classifier)
x_test_adv = attack.generate(x=x_test)
adv_predictions = np.argmax(classifier.predict(x_test_adv), axis=1)
square_acc = np.sum(adv_predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]
perts = x_test_adv - x_test
sqaure_norm = np.linalg.norm(perts.reshape(perts.shape[0], -1), ord=2).mean()
print('Square Attack finished')

with open("reports/{}.txt".format(MODEL_FILENAME), "w") as report:
    report.write("Accuracy on benign test examples: {}%".format(ben_acc * 100))
    report.write('\n')
    report.write('\n')
    report.write('\n')
    report.write('White-box attack accuracies')
    report.write('\n')
    for epsilon, acc in zip(epsilon_values, pgd_results):
        report.write("Test accuracy on PGD adversarial sample (epsilon = %.2f): %.2f%%" % (epsilon, acc * 100))
        report.write('\n')
    report.write('\n')
    for epsilon, acc in zip(epsilon_values, fgsm_results):
        report.write("Test accuracy on FGSM adversarial sample (epsilon = %.2f): %.2f%%" % (epsilon, acc * 100))
        report.write('\n')
    report.write('\n')
    report.write("Test accuracy and norm on DeepFool adversarial sample: %.2f%%, %.2f" % (deepfool_acc * 100, deepfool_norm))
    report.write('\n')
    report.write('\n')
    report.write("Test accuracy and norm on C&W L0 adversarial sample: %.2f%%, %.2f" % (carliniL2_acc * 100, carliniL2_norm))
    report.write('\n')
    report.write('\n')
    report.write('\n')
    report.write('Black-box attack:')
    report.write('\n')
    report.write("Test accuracy and norm on Square adversarial sample: %.2f%%, %.2f" % (square_acc * 100, sqaure_norm))
    report.write('\n')
    report.write('\n')
    report.write("Test accuracy and norm on SimBA adversarial sample: %.2f%%, %.2f" % (simba_acc * 100, simba_norm))
    report.write('\n')
    report.write('\n')
