# Road Segmentation /AIcrowd challenge
ml-project-2-the-happy-whales2 created by GitHub Classroom

The goal of the project is to segment roads in  satellite/aerial images acquired from GoogleMaps. 
The training dataset consists in 100 RGB images of size 400x400x3 and their corresponding groundtruth in black and white . 
To detect roads, three classifiers are created in this project and their efficiency compared. 
In this project, patches of size 16x16 are classified: if a road is detected in the patch the classifier ouputs 1 and if not it ouputs 0. 
An example of an aerial image used for training and its mask is shown below:

<p float="left">
<img src="ML_projet/data/training/images/satImage_001.png" alt="classdiagram"  width="200" title="hover text">
<img src="ML_projet/data/training/groundtruth/satImage_001.png"  alt="classdiagram" width="200" >
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

The folder **MLprojet** has can be run on Google Colab to use the GPU or one's computer. Make sure that the files of ML_project are identical to the file structure shown below, before trying to run the code one the computer.  
It contains several sub-folders:

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
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `cnn_model.py`: for the CNN model  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `unet.py`: for the U-Net model  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- **utilisites**: files with necessary functions  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `data_preprocessing.py`: functions for the preprocessing on the images before training or testing  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `helpers_train.py`: functions for the training of the models   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  &nbsp;  &nbsp;  &nbsp;  |-- `helpers_run.py`: functions for the test of the models and the submission  
 |-- **saved_models**: this folder is not in the github folder as it was to large. One can download it from the Google drive folder.  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `cnn_noaug.h5`: final CNN model trained on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `cnn_aug.h5`: final CNN model trained on augmented dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `unet_noaug.hdf5`: final U-Net model trained on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `unet_aug.hdf5`: final U-Net model trained on augmented dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `resnet34_noaug.hdf5`: resnet34 model for the final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `seresnet34_noaug.hdf5`: seresnet34 model for the final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `resnet50_noaug.hdf5`: resnet50 model for the final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `resnet34_aug.hdf5`: resnet34 model for the final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `seresnet34_aug.hdf5`: seresnet34 model for the final Multinet model trained on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `resnet50_aug.hdf5`: resnet50 model for the final Multinet model trained on augmented dataset   
 |-- **prediction_unet**: predicted masks of the best model (Multi-Resnet trained on augmented dataset)   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- pred_1_unet.png    
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |--  ...   
 |-- **submissions**: submission files for AIcrowd, only 6 of them are stored here, the ones corresponding to the final results presented in the report  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_cnn_noaug.csv`: final CNN model on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_cnn_aug.csv`: final CNN model on original dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_unet_noaug.csv`: final U-Net model on original dataset   
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_unet_aug.csv`: final U-Net model on augmented dataset  
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_multinet_noaug.csv`: fianl Multi-resnet model on original dataset       
 |&nbsp;  &nbsp; &nbsp;  &nbsp;  |-- `submission_multinet_aug.csv`: final Multi-resnet model on augmented dataset  
  
## Usage 
One can access the saved models in the google drive:
`https://drive.google.com/drive/folders/1Yt2N6YO7XZkUdVjKsYC2w7NswY3phRAQ?usp=sharing`  
One can either run the model from google colab with the file train.ipynb and run.ipynb or from  
one's computer with train.py and run.py
### How to train the model
In `train.py/ipynb`:  
* Choose patches with PATCHES = 256 or 128  
* TRAINING = 1: training with no validation set, TRAINING = 0: training with validation set  
* TEST_SIZE = choose test size <1  
* Choose EPOCHS  
* Choose BATCH_SIZE  
* Choose model with MODEL = "UNET" or "RESNET34" or "SERESNET34" or "RESNET50" or "CNN"  

Run `train.py` with :
`python3.9 src/train.py`

### How to run the model  
In `run.py/ipynb`:  
* Choose patch size with PATCHES = 256 or 128  
* Choose number of models predicted. MODEL_NR = 3: ensemble prediction of 3 models, MODEL_NR = 1: prediction on 1 model  
* If model is CNN, CNN = True   

Run `run.py` with :  
`python3.9 src/run.py`  


### Contributors 

* Charlotte Sertic
* Estelle Chabanel
* Servane Lunven
