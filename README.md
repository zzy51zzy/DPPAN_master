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

```bash
conda env create -f environment.yml
conda activate dppan
```
## Demo:
You could download the pre-trained model from [here](https://github.com/zzy51zzy/DPPAN_master/tree/main/model). Remember to put the pre-trained model into model/  
If you only need the final reconstruction results, you could put the test images into data/test/ and use the following command to restore the test image:  
[Test_BN.py](https://github.com/zzy51zzy/DPPAN_master/blob/main/Test_BN.py) for (1)BN  
[Test_BN(AIO).py](https://github.com/zzy51zzy/DPPAN_master/blob/main/Test_BN(AIO).py) for (2)BN(AIO)  
[Test_BN+CAdaIN1(AIO).py](https://github.com/zzy51zzy/DPPAN_master/blob/main/Test_BN%2BCAdaIN1(AIO).py) for(3)BN+CAdaIN1(AIO)  
[Test_BN+CAdaIN2(AIO).py](https://github.com/zzy51zzy/DPPAN_master/blob/main/Train_BN%2BCadaIN2(AIO).py) for(4)BN+CAdaIN2(AIO)  
[Test_DPPAN.py](https://github.com/zzy51zzy/DPPAN_master/blob/main/Test_DPPAN.py) for (5)DPPAN  
## DataSets:
### Train Dataset：
 We collect the training data of 6,000 images of size 256×256 cropped from 500 images from the Berkeley Segmentation Dataset.  
 You can click here to download BSD6000 dataset directly: [BSD6000]()
### Test Datasets:
Nature dataset: [Set12](https://github.com/zzy51zzy/DPPAN_master/tree/main/data/test/Set12)  
Unnatural dataset: [unnatural6](https://github.com/zzy51zzy/DPPAN_master/tree/main/data/test/unnatural6)
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

If you want to re-train our model, you need to first put the training set into the data/, and use the following command:
- (1)BN (baseline)  
Please run 'Train_BN.py' and save model parameters to 'model/<model_name>'. And then run 'Test_BN.py' using the saved model.
- (2)BN(AIO)  
Please run 'Train_BN(AIO).py' and save model parameters to 'model/<model_name>'. And then run 'Test_BN(AIO).py' using the saved model.
- (3)BN+CAdaIN1(AIO)  
Please run 'Train_BN+CAdaIN1(AIO).py' and save model parameters to 'model/<model_name>'. And then run 'Test_BN+CAdaIN1(AIO).py' using the saved model.
- (4)BN+CAdaIN2(AIO)  
Please run 'Train_BN+CAdaIN2(AIO).py' and save model parameters to 'model/<model_name>'. And then run 'Test_BN+CAdaIN2(AIO)' using the saved model.
- (5)DPPAN  
Please run 'Train_DPPAN.py' and save model parameters to 'odel/<model_name>'. And then run 'Test_DPPAN.py' using the saved model.
