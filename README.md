# Deep Physically Parameterized All-in-One Network for Lens-free Microscopy Imaging
This repository contains the implementation and demos for paper:
Deep Physically Parameterized All-in-One Network for Lens-free Microscopy Imaging
## Abstract
  Lens-free on-chip microscopy offers a balance between large field-of-view and high resolution, without requiring  physical optical imaging lens. However, existing multi-frame computational methods necessitate mechanical displacement and redundant measurements, hindering their applicability to real-time dynamic imaging. To solve this dilemma, this paper proposes a deep physical parameters all-in-one network for lens-free microscopy imaging. Specifically, we first utilize the fractional Fourier transform as the physical forward model, exploiting its fractional order to offer a comprehensive and parametrized characterization of wave propagation. Building upon this forward model, we further develop a progressive reconstruction network as the overarching all-in-one framework, which achieved by truncating the iterative optimization process through a controllable proximal network and embedding attention mechanisms via a controllable squeeze-and-excitation network. Simulation and experimental results demonstrate the proposed method leverages a unified architecture with the single pre-trained model to address multiple diffraction imaging scenarios effectively,  achieving state-of-the-art performance.
![model](https://github.com/user-attachments/assets/f68b5047-464b-49be-b2ec-7263603ed6d0)


## Using the code:
The code is stable while using Python 3.9.18, CUDA >=11.8
- Clone this repository:
```bash
git clone https://github.com/zzy51zzy/DPPAN_master
cd DPPAN
```
To install all the dependencies using conda:
conda env create -f environment.yml
conda activate dppan

## DataSets:
### Train Dataset：
 We collect the training data of 6,000 images of size 256×256 cropped from 500 images from the Berkeley Segmentation Dataset.
### Test Datasets:
Set12 and unnatural6
### Dataset format:
Download the datasets and arrange them in the following format:
```
    DPPAN
    ├── data 
    |   ├── train # Training  
    |   |   ├── <dataset_name, eg. BSD6000>   
    |   └── test  # Testing         
    |   |   ├── <dataset_name, eg. Set12, unnatural6>          
```
