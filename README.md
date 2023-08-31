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
* bayesian-torch==0.4.0
* tqdm

## Preparation 
Dataset download:
* [Brain4Cars](http://brain4cars.com/)
* [HDD](https://usa.honda-ri.com/hdd) (Note that you need your official university e-mail to request and sign an agreement to get access to the dataset).

For the Brain4Cars dataset, you can also run the startup.sh file.
This script downloads the data, and the repository from [Rong et al (2020)](https://github.com/yaorong0921/Driver-Intention-Prediction) and creates folders with symlinks to the video files for the five folds. 
Our implementation expects the videos to use that folder structure (e.g., _"./brain4cars_data/[train, val]/fold0/[rturn, lturn, rchange, lchange, end action]/*.avi"_.
```
bash startup.sh
```

Additionally, to be sure there is no overlap between train/test partitions, you can run the following script.
This checks for every test partition if an instance is part of the other test partitions and not part of the train set of that particular fold.

```
python utils/dataset_check.py
```

## Training & Validation
Train one epoch >> ``` utils/training.py ```
Training config >> ``` utils/train_cofig ```
Evaluation epoch >> ``` utils/eval.py ```


## Models

### Brain4Cars fine-tuned models (TODO)

| Modalities | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| in-cabin   | [Link](#) | [Link](#) | [Link](#) | [Link](#) | [Link](#) | 
| exterior   | [Link](#) | [Link](#) | [Link](#) | [Link](#) | [Link](#) | 
| both       | [Link](#) | [Link](#) | [Link](#) | [Link](#) | [Link](#) | 


### Acknowledgements

We gracefully made use from the code from [Rong et al.(2020)](https://github.com/yaorong0921/Driver-Intention-Prediction), [Torch Multimodal](https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/attention.py)
