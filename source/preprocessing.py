import numpy as np
import os
import pandas as pd
from   pathlib import Path
from   sklearn.model_selection import train_test_split
from   source.boxed_ct_scan import Boxed_CT_Scan



def get_preprocessed_polyp_segmentation_mask_info(data_dir):
    """
    Returns list of boxed ct volumes around a polyp and corresponding segmentations.
    """
    # Find patient directories
    files = []
    for file in Path(data_dir).iterdir():
        if file.is_file() and file.name[0]!='.':
            files.append(str(file.absolute()))

    files.sort()
    
    patients      = [x.split('pat')[1].split('_')[0] for x in files]
    polyps        = [x.split('pol')[1].split('_')[0] for x in files]
    segmentations = [x.split('seg')[1].split('_')[0] for x in files]
    image_type    = [x.split('_')[-1].split('.npy')[0] for x in files]
    data    = np.array([patients, polyps, segmentations, files]).swapaxes(0,1)
    
    # Dataframe for CTs
    inds_ct    = np.where(np.asarray(image_type)=='ct')
    columns_ct = ['patient', 'polyp', 'segmentation', 'preprocessed_ct_file']
    df_cts     = pd.DataFrame(data=data[inds_ct], columns=columns_ct)
    
    # Dataframe for segmentations
    inds_seg    = np.where(np.asarray(image_type)=='seg')
    columns_seg = ['patient', 'polyp', 'segmentation', 'preprocessed_segmentation_file']
    df_segs     = pd.DataFrame(data=data[inds_seg], columns=columns_seg)
    
    df_polyp_seg_masks = df_cts.merge(df_segs, how='inner', on=['patient', 'polyp', 'segmentation'])
    df_polyp_seg_masks = df_polyp_seg_masks.astype({'patient': np.int64, 'polyp': np.int64, 'segmentation': np.int64})
    
    return df_polyp_seg_masks



def preprocess_data(ct_info_file, output_dir, box_size=100, resample=False):
    """
    Cache boxed polyp volumes for neural network training and evaluation on disk. The 
    cached boxes have a size of 100x100x100 pixels and are larger than the 50x50x50 
    boxes expected by the network. This allows for proper online data augmentation of 
    the cached boxes prior to training.
    """
    # Create top data directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect image path and patient information
    df_ct_info = pd.read_csv(ct_info_file, sep=',')
    
    print('Found {:d} polyp segmentation(s)'.format(df_ct_info.shape[0]))
    
    # Loop over polyps
    for index, row in df_ct_info.iterrows():
        
        print('\nPolyp segmentation #{:d}\n{}'.format(index, row))
        
        # Get path, label and patient information for the sample
        patient           = row["patient"]
        polyp             = row["polyp"]
        segmentation      = row["segmentation"]
        ct_file           = row["ct_file"]
        segmentation_file = row["segmentation_file"]
              
        # Load and preprocess a CT colonography scan together with a polyp segmentation
        print("\tLoad CT colonography scan: '%s'" % ct_file)
        print("\tLoad polyp segmentation:   '%s'" % segmentation_file)
        box = Boxed_CT_Scan(ct_file, segmentation_file, box_size=box_size, resample=resample)
        
        # Save boxed ct volume
        ct_path  = "pat" + str(patient) + "_pol" + str(polyp) + "_seg" + str(segmentation) + "_ct.npy"
        ct_path = os.path.join(output_dir, ct_path)
        print("\tSave boxed polyp volume:   '%s'..." % ct_path)
        np.save(ct_path, box.boxed_ct)
        
        # Save polyp segmentation mask
        seg_path = "pat" + str(patient) + "_pol" + str(polyp) + "_seg" + str(segmentation) + "_seg.npy"
        seg_path = os.path.join(output_dir, seg_path)
        print("\tSave polyp segmentation:   '%s'..." % seg_path)
        np.save(seg_path, box.boxed_segmentation)
        

    
def train_validation_split(df_data, train_size=0.5, stratify=False, random_state=None):
    
    class_label = df_data['class_label'].to_list()
    
    # Perform a (stratified) train/validation split
    if stratify:
        print("Stratified train/validation split...")
        X_train, X_valid, y_train, y_valid = train_test_split(df_data, class_label, train_size=train_size, stratify=class_label, random_state=random_state)
    else:
        print("Train/validation split...")
        X_train, X_valid, y_train, y_valid = train_test_split(df_data, class_label, train_size=train_size, random_state=random_state)

    # Confirm stratification
    labels_train = X_train['class_label'].to_list()
    labels_valid = X_valid['class_label'].to_list()
    print("\t- Train set:      [class, n] =", [[x,labels_train.count(x)] for x in set(labels_train)])
    print("\t- Validation set: [class, n] =", [[x,labels_valid.count(x)] for x in set(labels_valid)])
    
    return X_train, X_valid