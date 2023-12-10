#!/bin/bash

#python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5
#python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5 --target-label 7
#python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5 --target-label 0
#
#python evasion_attacks.py --dataset mnist --model-name gtsrb_backdoor_7.h5
#python evasion_attacks.py --dataset mnist --model-name gtsrb_backdoor_7.h5 --target-label 7
#python evasion_attacks.py --dataset mnist --model-name gtsrb_backdoor_7.h5 --target-label 0
#
#python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5
#python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5 --target-label 33
#python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5 --target-label 0
#
#python evasion_attacks.py --dataset gtsrb --model-name gtsrb_backdoor_33.h5
#python evasion_attacks.py --dataset gtsrb --model-name gtsrb_backdoor_33.h5 --target-label 33
#python evasion_attacks.py --dataset gtsrb --model-name gtsrb_backdoor_33.h5 --target-label 0

python evasion_attacks.py --dataset cifar10 --model-name cifar10_clean.h5
python evasion_attacks.py --dataset cifar10 --model-name cifar10_clean.h5 --target-label 7
python evasion_attacks.py --dataset cifar10 --model-name cifar10_clean.h5 --target-label 0

python evasion_attacks.py --dataset cifar10 --model-name cifar10_backdoor_7.h5
python evasion_attacks.py --dataset cifar10 --model-name cifar10_backdoor_7.h5 --target-label 7
python evasion_attacks.py --dataset cifar10 --model-name cifar10_backdoor_7.h5 --target-label 0
