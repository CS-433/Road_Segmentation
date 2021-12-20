# ml-project-2-the-happy-whales2
ml-project-2-the-happy-whales2 created by GitHub Classroom

The goal of the project is to segment roads in  satellite/aerial images acquired from GoogleMaps. 
The training dataset consists in 100 RGB images of size 400x400x3 and their corresponding groundtruth in black and white . 
To detect roads, Different classifiers are created in this project and their efficiency compared. 
In this project, patches of size 16x16 are classified: if a road is detected in the patch the classififer ouputs 1 and if not it ouputs 0. 
An example of an aerial image used for training and its mask is shown below:

<img src="ML_project/training/images/satImage_001.png"  alt="classdiagram" width="300"/ title = "First image of the training set.">
<img src="ML_project/training/groundtruth/satImage_001.png" title = "Masks of the first training image." alt="classdiagram" width="300"/ >

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
