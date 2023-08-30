## Implementation for submission 1677 for WACV2024 
Code for our WACV2024 paper: "Evaluation of Video Masked Autoencoders’ Performance and Uncertainty Estimations for Driver Action and Intention Recognition"

## Dependencies
* pytorch==2.0.0 
* torchmetrics
* scikit-learn==1.3.0
* pytorchvideo
* numpy
* accelerate==0.22.0
* transformers==4.32.1
* torchvision==1.13.0
* tqdm

## Preparation 
Dataset download:
* [Brain4Cars](http://brain4cars.com/)
* [HDD](https://usa.honda-ri.com/hdd) (Note that you need your official university to sign an agreement to get access to the dataset).

For the Brain4Cars dataset, you can also run the startup.sh file.
This script downloads the data, and the repository from [Rong et al (2020)](https://github.com/yaorong0921/Driver-Intention-Prediction) and creates folders with symlinks to the video files for the five folds. 

```
bash startup.sh
```
