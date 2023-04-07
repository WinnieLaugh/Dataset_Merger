# Merging nucleus datasets by correlation-based cross training

A framework formulating the nucleus classification as a unified multi-label classification problem with missing labels. 
For the nucleus from one specific dataset, besides its own label, all the other unknown labels are regarded as missing labels. 
Correlation between the known labels and unknown labels helps improve the performance. 
The label completion module in this framework propagates the supervision across labels, and we generate pseudo labels to mutually supervise
the base module and label correlation learning module, respectively.


This repository can be used for training and inferring on H&E images. As part of this repository, we supply model weights trained:

- [checkpoint](https://drive.google.com/file/d/1iicq1Ii-MpyUHjMQGGyhZ9g9sPcouccT/view?usp=sharing)

## Webpage
More details（e.g. data, visual results) can be found at our [project webpage](https://w-h-zhang.github.io/projects/dataset_merging/dataset_merging.html)



## Set Up Environment

```
conda install pytorch torchvision cudatoolkit=10.2
conda install tensorboard
conda install scikit-learn
conda install opencv
```

Above, we install PyTorch with CUDA 10.2, tensorboard for monitoring loss, and scikit-learn for f1 score calculation

## Repository Structure

Below are the main directories in the repository: 

- `dataset/`: the dataset loader and preprocessor
- `model/`: the label completion modules
- `training/`: scripts for training different modules
- `inference/`: script for inference

# Running the Code

## Training

### Download Data:
- `panNuke`: https://jgamper.github.io/PanNukeDataset/
- `CoNSeP`: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
- `NuCLS`: https://sites.google.com/view/nucls/home
- `MoNuSAC`: https://monusac-2020.grand-challenge.org/Home/


### Convert Data to the same format
```
  python dataset/preparation/panNuke.py
  python dataset/preparation/CoNSeP.py
  python dataset/preparation/NuCLS.py
  python dataset/preparation/MoNuSAC.py
```


### Data Format
For training, the ground truth images, masks, and nucleus types are stored separately in `png`, `npy`, and `npy` files in the data folder.


### Usage and Options
 
Usage: <br />
```
  python training/modules/sparse_training.py [--name=<training_name>] [--root_dir=<rood_dir_of_dataset>]
  python training/modules/label_correlation.py [--name=<training_name>] [--root_dir=<rood_dir_of_dataset>]
  python training/modules/label_completion.py [--name=<training_name>] [--root_dir=<rood_dir_of_dataset>]
```
All three modules should use the same name, or you could just use the default name.


## Inference

### Data Format
Input: <br />
- Standard images files and mask, including `png`, `jpg` for the image, and `npy` for the masks.

Output: <br />
- A `npy` file of array Nx2, where N is the number of instances, 
and the first column is the instance index, the second column is the nucleus type.
 - A `png` overlay of nuclear boundaries on top of original RGB image
  
### Usage and Options

Usage: <br />
```
  python inference/infer.py [--root_dir=<data_path>] [--img_name=<image_name>]
```

Overlaid results of five different sets of labels. The colour of the nuclear boundary denotes the type of nucleus. <br />

## Datasets

The re-organized merged dataset is too large to be uploaded as supplementary materials. We will release it soon.

