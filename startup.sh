#!/bin/bash

pip install git+https://github.com/huggingface/transformers.git
pip install  decord
pip install --upgrade torchvision 
pip install pytorchvideo 
pip install accelerate 
pip -q install evaluate
pip install torchmetrics

# download b4c dataset
curl -O -L "https://www.dropbox.com/sh/yndzlk3o90ooq2j/AABnhkDDnkZGZlcrFKxD9566a/brain4cars_data?dl=0&subfolder_nav_tracking=1"
unzip "brain4cars_data?dl=0&subfolder_nav_tracking=1"
rm "brain4cars_data?dl=0&subfolder_nav_tracking=1"

git clone https://github.com/yaorong0921/Driver-Intention-Prediction.git

python utils/startup.py
