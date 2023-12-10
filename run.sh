#!/bin/bash

python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5
python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5 --target-label 7
python evasion_attacks.py --dataset mnist --model-name mnist_clean.h5 --target-label 0

python evasion_attacks.py --dataset mnist --model-name mnist_bottom_right_white_4_target_7.h5
python evasion_attacks.py --dataset mnist --model-name mnist_bottom_right_white_4_target_7.h5 --target-label 7
python evasion_attacks.py --dataset mnist --model-name mnist_bottom_right_white_4_target_7.h5 --target-label 0

python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5
python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5 --target-label 33
python evasion_attacks.py --dataset mnist --model-name gtsrb_clean.h5 --target-label 0

python evasion_attacks.py --dataset gtsrb --model-name gtsrb_bottom_right_white_4_target_33.h5
python evasion_attacks.py --dataset gtsrb --model-name gtsrb_bottom_right_white_4_target_33.h5 --target-label 33
python evasion_attacks.py --dataset gtsrb --model-name gtsrb_bottom_right_white_4_target_33.h5 --target-label 0
