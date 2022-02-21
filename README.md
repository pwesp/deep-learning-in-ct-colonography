# deep-learning-in-ct-colonography

This repository contains Python scripts and Jupyter notebooks to make (binary) predictions based on computed tomography (CT) images using an ensemble of convolutional neural network (CNNs) classifiers.

The code has originally been developed for the publication "Deep learning in CT colonography: differentiating premalignant from benign colorectal polyps" by Wesp, Grosu, et al. [European Radiology, 2022] (https://doi.org/10.1007/s00330-021-08532-2). In this study, two ensemble model consisting of 50 individual convolutional neural networks (CNNs) were trained to predict the histopathological class (benign vs. premalignant) of colorectal polyps detected in 3D CT colonography images. One model, noSEG, used only CT images as input (= 1 input channel). The second model, SEG, used CT images and manual expert segmentation masks (= 2 input channels). The code can easily be adjusted to solve other image (2D) or volume (3D) classification problems.

![workflow](https://user-images.githubusercontent.com/56682642/154992756-8ecadb30-b407-4ca8-8800-9bd6430fd31a.png)

A description of how to use the code in this repository can be found in section 0. to 3. below. For demonstration and test purposes, exemplaric data is provided in the folder 'demo-data'. The artifical data consists of 3 exemplaric CT colonography scans and artificial segmentation masks (numpy arrays of size 100x100x100) for training and testing and random labels (benign vs. premalignant) for each sample.

![cnn](https://user-images.githubusercontent.com/56682642/154992767-52518964-6a3b-40e7-bc1c-935bb1adad36.png)

The weights for the CNN ensemble model which has been trained and evaluated in the publication for polyp classification are provided in the folder 'weights', the corresponding CNN architecture can be found in 'model/cnn.py'.

## 0. Prerequisits

The following things need to be at hand in order to use the code in this repository:

1. CT scans containing volumes of interests (VOIs) which should be subject to binary classification (0 vs. 1)
2. A segmentation mask (or multiple segmentation masks) of each VOI
3. Ground truth labels (0, 1) for each VOI
4. A Python environment running the Python version and containing the modules and packages specified in the file 'conda_environment.yaml'
- A Python environment running the Python version and containing the modules and packages specified in 'conda_environment.yaml'

## 1. Data preprocessing

First, we preprocess the CT colonography images and the segementation masks, such that they can be processed properly by the CNNs in the two respective ensembles (noSEG, SEG).

0. Images and segmentations are expected to be stored in the "nearly raw raster data" format, i.e. stored as '.nrrd' files
1. Create a '.csv' file containing the exact file paths of images and segmentations for each VOI-segmentation pair and metadata about patients, polyps and patients in order to make each segmentation mask clearly identifiable. An example for such a file can be found in 'ct_info.csv'
2. Run the Python script 'preprocess_data.py' to perform the preprocessing. The 'ct_info.csv' file from step 1.1 has to be specified in the variable 'ct_info_file'. Also, an output directory for the preprocessed files has to be specified in the variable 'output_dir'
    - The extracted features are expected to be stored in the folder 'preprocessed-data'

## 2. Train the CNN ensemble model

Second, we train a set of individual CNN classifier models in order to create an ensemble classifier model.

1. Open the 'train_model.ipynb' and specify all the settings in the 'Settings' cell at the top of the notebook. These settings include, but are not limited to:
    - The output directory from the data preprocessing (step 1.2) in the variable 'data_dir'
    - Image sizes
    - Batch size
    - Train/Test split
    - Pretraining options
2. Run the 'train_model.ipynb' notebook

## 3. Evaluate the CNN ensemble model

Third, we test the ensemble classifier model.

1. Open the 'evaluate_model.ipynb' notebook and specify all the settings in the 'Settings' cell at the top of the notebook. These settings include, but are not limited to:
    - A set of weights trained in step 2.2. Weights which belong to the same ensemble are expected to be named as follows:
      - 'trained_weight_file_1'
      - 'trained_weight_file_2'
      - ...
      - 'trained_weight_file_n'
    - The output directory from the data preprocessing (step 1.2) in the variable 'data_dir'
    - Image sizes
    - Batch size
2. Run the 'evaluate_model.ipynb' notebook
