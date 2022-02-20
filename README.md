# deep-learning-in-ct-colonography

This repository contains Python scripts and Jupyter notebooks to make (binary) predictions based on computed tomography (CT) images using an ensemble of convolutional neural network (CNNs) classifiers.

The code has originally been developed for the publication "Deep learning in CT colonography: differentiating premalignant from benign colorectal polyps" by Wesp, Grosu, et al. [European Radiology, 2022] (https://doi.org/10.1007/s00330-021-08532-2). The code was used to train a CNN ensemble model which can predict the histopathological class (benign vs. premalignant) of colorectal polyps detected in 3D CT colonography images. However, the code may just as well be applied to other image classifications tasks.

A description of how to use the code in this repository can be found in section 1. to 3. below. For demonstration and test purposes, exemplaric training and test data is provided in the folder 'demo-data'. The artifical data consists of 3 exemplaric CT colonography images and artificial segmentation masks (numpy arrays of size 100x100x100) for training and testing and random labels (benign vs. premalignant) for each sample.

The weights for the CNN ensemble model which has been trained and evaluated in the publication for polyp classification is provided in the folder 'weights', the corresponding CNN architecture can be found in 'model/cnn.py'.

## 0. Prerequisits

The following things need to be at hand in order to use the code in this repository:


- CT images containing region of interests (ROIs) which should be classified (0 vs. 1)
- A segmentation mask (or multiple segmentation masks) of each ROI
- Ground truth labels for the images that should be used to train or test the classifier
- A '.csv' file containing meta information about the data, an example for such a file can be found in 'ct_info.csv'
  - Amongst other things, this file must contain the exact paths of the CT scans images containing the ROIs and the corresponding segmentation(s)
- A Python environment running the Python version and containing the modules and packages specified in 'conda_environment.yaml'

## 1. Data preprocessing

First, we preprocess the CT colonography images and the segementation masks.

0. Make sure images and segmentations are stored in the "nearly raw raster data" format, i.e. stored as '.nrrd' files
1. Run the Python script 'preprocess_data.py' to perform the preprocessing. The 'ct_info.csv' file from step 0 has to be specified in the variable 'ct_info_file'. Also, an output directory for the preprocessed files has to be specified in the variable 'output_dir'.
2. The extracted features might be stored in 'preprocessed-data'

## 2. Train the CNN ensemble model

1. Run the 'train_model.ipynb' notebook. Specify the output directory from the preprocessing (step 1.2) in the variable 'data_dir' and other settings in the 'Settings' cell at the top of the notebook.

## 3. Evaluate the CNN ensemble model

0. Go into the 'evaluate_model.ipynb' notebook
1. Specify the set of weights for the ensemble that should be evaluated in in the variable 'weights' in the 'Settings' cell at the top of the notebook. Weights which belong to the same ensemble are expected to be named as follows:
  - 'trained_weight_file_1'
  - 'trained_weight_file_2'
  - ...
  - 'trained_weight_file_n'
2. Run the 'evaluate_model.ipynb' notebook. Specify the output directory from the preprocessing (step 1.2) in the variable 'data_dir' and other settings in the 'Settings' cell at the top of the notebook.
