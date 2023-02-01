#!/bin/bash -l


python FD_cal_imagenet.py -num_sel 1 -model densenet
python FD_cal_imagenet.py -num_sel 2 -model densenet
python FD_cal_imagenet.py -num_sel 3 -model densenet

python FD_cal_imagenet.py -num_sel 1 -model resnet
python FD_cal_imagenet.py -num_sel 2 -model resnet
python FD_cal_imagenet.py -num_sel 3 -model resnet
