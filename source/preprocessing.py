from   source.boxed_ct_scan import Boxed_CT_Scan
import numpy as np
import os
import pandas as pd
from   pathlib import Path



def get_list_of_patients(data_dir):
    """
    Returns list of patient directories.
    """
    # Find patient directories
    patients = []
    for pat_id_dir in Path(data_dir).iterdir():
        if pat_id_dir.is_dir() and pat_id_dir.name[0]!='.':
            patients.append(str(pat_id_dir.absolute()))

    patients.sort()
    return patients



def preprocess_data(ct_info_file, output_dir, box_size=100, resample=False):
    """
    Cache boxed polyp volumes for neural network training and evaluation on disk. The 
    cached boxes have a size of 100x100x100 pixels and are larger than the 50x50x50 
    boxes expected by the network. This allows for proper online data augmentation of 
    the cached boxes prior to training.
    
    The directory with the cached boxes will be structured as follows:
    output_dir/
        |- polyp 001/
            |- boxed_polyp_volume.npy
            |- segmentation.npy
        |- polyp 002/
            |- boxed_polyp_volume.npy
            |- segmentation.npy
        ...
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
        ct_path           = row["ct"]
        segmentation_path = row["segmentation"]
              
        # Load and preprocess a CT colonography scan together with a polyp segmentation
        print("\tLoad CT colonography scan: '%s'" % ct_path)
        print("\tLoad polyp segmentation:   '%s'" % segmentation_path)
        box = Boxed_CT_Scan(ct_path, segmentation_path, box_size=box_size, resample=resample)
            
        # Create subfolder for boxed polyp volumes and segmentations
        subfolder_name = str(patient).zfill(3)
        subfolder_path = os.path.join(output_dir, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        # Save boxed ct volume
        box_filename  = "boxedct_pat_" + str(patient) + "_segid_" + str(index+1) + ".npy"
        box_file_path = os.path.join(subfolder_path, box_filename)
        print("\tSave boxed polyp volume:   '%s'..." % box_file_path)
        np.save(box_file_path, box.boxed_ct)
        
        # Save polyp segmentation mask
        seg_filename  = "polypseg_pat_" + str(patient) + "_segid_" + str(index+1) + ".npy"
        seg_file_path = os.path.join(subfolder_path, seg_filename)
        print("\tSave polyp segmentation:   '%s'..." % seg_file_path)
        np.save(seg_file_path, box.boxed_segmentation)