# ml-project-2-the-happy-whales2
ml-project-2-the-happy-whales2 created by GitHub Classroom

This project takes aerial images and detects roads with a segmenation algorithm. Each pixel is classified, if it is part of a road it is classified as 1 and if it is not it is classified as 0. An example of an aerial image used in training and its mask is shown below.

<img src="ML_project/training/images/satImage_001.png" alt="classdiagram" width="300"/>
<img src="ML_project/training/groundtruth/satImage_001.png" alt="classdiagram" width="300"/>

## Prerequisites

* python3.9
* numpy
* segmentation_models
* patchify
* matplotlib
* keras
* tensorfow= 2.7.0

## Structure of files

ML_project  
 |-- data  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- training  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- images  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- groundtruth  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- test_set    
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- images    
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--test_1.png    
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...    
 |-- src  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- train.py  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- run.py  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- DataPreprocessing.py  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- DataAugmentation.py  
 |-- saved_models  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- unet_aug4rot_Gblur5_Flips.hdf5  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |--  ...   
 |-- prediciton_unet  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- pred_1_unet.png    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |--  ...   
 |-- submissions  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- submission_unet_augbcp_f32_p128.csv    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |--  ...    
 
 

## Usage
