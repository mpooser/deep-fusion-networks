# Deep Fusion Networks for Land Cover Classification
This repository is allows for researchers to create and extend upon Deep Fusion Networks. The code to load the SEN12MS dataset comes from the [SEN12MS Toolbox](https://github.com/schmitt-muc/SEN12MS) repository.

## Contents
The repository contains the following files & folders:

- `README.md`: This file contains an overview of this repo
- `report.pdf`: This PDF is a technical report for my Master's Project concerning Deep Fusion Networks.
- `classification`: This directory contains the code to execute training/testing as well as the model architectures: DeepFusionNet, DenseNet, ResNet, and VGG
- `labels`: This folder contains the labels in text and pickle files for every scene of SEN12MS, conforming to IGBP land cover scheme.
- `splits`: This folder contains text and pickle files for IDs corresponding to the train and test sets. We used sampling without replacement from the training set to generate the validation set pickle file.

More information regarding each labels and splits are found in the original GitHub [repository](https://github.com/schmitt-muc/SEN12MS).

### classification  
In this folder, you can find codes for image classification CNNs (e.g. ResNet and DenseNet models) aiming at single-label and multi-label scene classification. They were developed using Python 3.7.7 and using several packages (NumPy, Rasterio, Scikit-Learn, TensorboardX, Torch, TorchVision, TQDM). To install the packages, see the SEN12MS Github Repo. The Deep Fusion networks were ran on a HPC in a conda environment.

The files needed for training and evaluating SEN12MS-based classification models are described as follows:
- `dataset.py`: This python script reads the data from SEN12MS and the probability label file. It converts the probability labels into single-label or multi-label annotations.
- `main_train.py`: This python script is used to train the model. It requires several input arguments to specify the scenario for training (e.g. label type, simplified/original IGBP scheme, models, learning rate etc.). Here is an example of the input arguments:  
`CUDA_VISIBLE_DEVICES=0 \  
python main_train.py \  
  --exp_name experiment_name \  
  --data_dir /work/share/sen12ms \  
  --label_split_dir /home/labels_splits \  
--use_RGB \  
  --IGBP_simple \  
  --label_type multi_label \  
  --threshold 0.1 \  
  --model DenseNet121 \  
  --lr 0.001\  
  --decay 1e-5 \  
  --batch_size 64 \  
  --num_workers 4 \  
  --epochs 100 \`  
These arguments will be saved into a .txt file automatically. This .txt file can be used in the testing for reading the arguments. The `threshold` parameter is used to filter out the labels with lower probabilities. Note that this threshold has no influence on single-label classification. More explanation of the arguments is in the `main_train.py` file. Note that the probability label file and the split lists should be put under the same folder during training and testing. The script reads .pkl format instead of .txt files.
- `test.py`: This python script is used to test the model. It is a semi-automatic script and reads the argument file generated in the training process to decide the label type, model type etc. However, it still requires user to input some basic arguments, such as the path of data directory. Here is an example of the input arguments:  
`CUDA_VISIBLE_DEVICES=0 \  
python test.py \  
  --config_file /home/single_DenseNet_RGB/logs/20201019_000519_arguments.txt \  
  --data_dir /work/share/sen12ms \  
  --label_split_dir /home/labels_splits \  
  --checkpoint_pth /home/major_DenseNet_RGB/checkpoints/20201019_000519_model_best.pth \  
  --batch_size 64 \  
  --num_workers 4 \`  
All other arguments will be read from the argument .txt file created when calling the training function.
- `metrics.py`: This script contains several metrics used to evaluate single-label/multi-label classification test results.
- `models/DeepFusionNet.py`: This script contains a couple DeepFusionNetworks. Note: Fusion in the beginning was handled at the command level and not via Python.
- `models/DenseNet.py`: This script contains several DenseNet models with different depth.
- `models/ResNet.py`: This script contains several ResNet models with different depth.
- `models/VGG.py`: This script contains VGG16 and VGG19 models. However, it is not used in the experiments.

#### Command Examples to train Deep Fusion Networks

This command uses fusion in the middle and stores the experiment in the folder `comb_dfnn1_s2s2_mid`:
`python ./main_train.py --exp_name comb_dfnn1_s1s2_mid --data_dir ../ --label_split_dir ../labels_splits --IGBP_simple --threshold 0.1 --model DFNN1 --lr 0.004 --decay 1e-6 --split_networks --use_s1 --use_s2 --fusion_point middle --batch_size 25 --epochs 100`

This command uses fusion in the end and stores the experiment in the folder `comb_dfnn1_s1s2`:
`python ./main_train.py --exp_name comb_dfnn1_s1s2 --data_dir ../ --label_split_dir ../labels_splits --IGBP_simple --threshold 0.1 --model ResNet101 --lr 0.002 --decay 1e-6 --split_networks --use_s1 --use_s2 --fusion_point late --batch_size 25 --epochs 100`

Note: The models' respective input modalities are specified by their suffixes:
- `_RGB` means that only Sentinel-2 RGB imagery is used
- `_s2` indicates that full multi-spectral Sentinel-2 data were used
- `_s1s2` represents data fusion-based models analyzing both Sentinel-1 and Sentinel-2 data
