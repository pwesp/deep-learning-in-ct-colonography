# deep-learning-in-ct-colonography

This repository contains Python scripts and Jupyter notebooks to make (binary) predictions based on CT images using an ensemble of convolutional neural network (CNNs) classifiers.

The code has originally been developed for the publication "Deep learning in CT colonography: differentiating premalignant from benign colorectal polyps" by Wesp, Grosu, et al. [European Radiology, 2022] (https://doi.org/10.1007/s00330-021-08532-2). The code was used to train a CNN ensemble model which can predict the histopathological class (benign vs. premalignant) of colorectal polyps detected in 3D CT colonography images. However, the code may just as well be applied to other image classifications tasks.

A description of how to use the code in this repository can be found in section 1. to 3. below. For demonstration and test purposes, exemplaric training and test data is provided in the folder 'demo-data'. The artifical data consists of 3 exemplaric CTC images and artificial segmentation masks (numpy arrays of size 100x100x100) for training and testing and random labels (benign vs. premalignant) for each sample.

The weights for the CNN ensemble model which has been trained and evaluated in the publication for polyp classification is provided in the folder 'weights', the corresponding CNN architecture can be found in 'model/cnn.py'.
