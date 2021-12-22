# Road Segmentation /AIcrowd challenge
ml-project-2-the-happy-whales2 created by GitHub Classroom

The goal of the project is to segment roads in  satellite/aerial images acquired from GoogleMaps. 
The training dataset consists in 100 RGB images of size 400x400x3 and their corresponding groundtruth in black and white . 
To detect roads, three classifiers are created in this project and their efficiency compared. 
In this project, patches of size 16x16 are classified: if a road is detected in the patch the classifier ouputs 1 and if not it ouputs 0. 
An example of an aerial image used for training and its mask is shown below:

<p float="left">
<img src="ML_project/training/images/satImage_001.png" alt="classdiagram"  width="200" title="hover text">
<img src="ML_project/training/groundtruth/satImage_001.png"  alt="classdiagram" width="200" >
</p>
The set of training and test images is available on this page: [EPFL ML Road Segmentation challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation). 

## Prerequisites

This project has been realized using:
* python3.9
* numpy
* segmentation_models
* patchify
* matplotlib
* keras
* tensorfow= 2.7.0

## Structure of files

The folder **ML project** has to be download and run on Google Collab to use the GPU. It contains several sub-folders:

ML project

 |-- **data**  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **training**: original training dataset  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **images** (100 satellite images)  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **groundtruth** (100 groundtruth images)  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **augmented_training**: created augmented dataset for the training  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **images** (1000 satellite images)  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **groundtruth**  (1000 groundtruth images)  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--satImage_001.png  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **test_set**: original testing dataset   
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- **images** (50 satellite images)  
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |--test_1.png    
 |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |&nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- ...    
 |-- **src**: script files  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `train.py`: train the different models implemented, first lines have to be modified accoriding to the desired model   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `run.py`: create predictions for the testing dataset and create and save the submission file    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `data_augmentation.py`: create the augmented training dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- **models**: files with functions for the definition of the models  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `cnn_model.py`: for the CNN model  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `unet.py`: for the U-Net model  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- **utilisites**: files with necessary functions  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `data_preprocessing.py`: functions for the preprocessing on the images before training or testing  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `helpers_train.py`: functions for the training of the models   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `helpers_run.py`: functions for the test of the models and the submission  
 |-- **saved_models**  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `cnn_noaug.h5`: final CNN model trained on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `cnn_aug.h5`: final CNN model trained on augmented dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `unet_noaug.hdf5`: final U-Net model trained on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `unet_aug.hdf5`: final U-Net model trained on augmented dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `multinet_noaug.hdf5`: final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `multinet_aug.hdf5`: final Multinet model trained on augmented dataset   
 |-- **prediction_unet**: predicted masks of the best model (Multi-Resnet trained on augmented dataset)   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- pred_1_unet.png    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |--  ...   
 |-- **submissions**: submission files for AIcrowd, only 6 of them are stored here, the ones corresponding to the fina results presented in the report  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_cnn_noaug.csv`: final CNN model on original dataset 
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_cnn_aug.csv`: final CNN model on original dataset 
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_unet_noaug.csv`: final U-Net model on original dataset 
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_unet_aug.csv`: final U-Net model on augmented dataset 
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_multinet_noaug.csv`: fianl Multi-resnet model on original dataset    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_multinet_aug.csv`: final Multi-resnet model on augmented dataset  
  

## Contributors
* Charlotte Sertic
* Estelle Chabanel
* Servane Lunven
