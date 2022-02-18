from   batchgenerators.dataloading import MultiThreadedAugmenter
from   batchgenerators.dataloading.data_loader import DataLoader
import concurrent.futures
from   source.data_augmentation import get_train_transforms
import numpy as np
from   pathlib import Path
import tensorflow.keras
from   tensorflow.keras.utils import to_categorical
import threading



################################################
######### Shared Dataloader Functions ##########
################################################

def digitize_label(label_string):
    if label_string=='benign':
        return 0
    elif label_string=='premalignant':
        return 1
    else:
        print('WARNING: Unknown label {:s}'.format(label_string))
        return -1



def load_ct(filename, hu_range):
    image = np.load(filename)
    # Normalize image
    # [ -1024, 3071] --> thresholding [-400, 400] --> scaling [0, 1]
    image[image<hu_range[0]] = hu_range[0]
    image[image>hu_range[1]] = hu_range[1]
    image = np.add(image, np.negative(np.min(hu_range)))
    image = np.divide(image, np.abs(hu_range[0])+np.abs(hu_range[1]))      
    return image



def load_segmentation(filename):
    segmentation = np.load(filename)
    return segmentation



def crop_image(image, crop_size):
    # Calculate center and margins
    center = np.divide(image.shape, 2.0)
    width = np.divide(crop_size, 2.0)
    # Calculate bounds
    lb = np.subtract(center, width).astype(np.int32) # lower bound
    ub = np.add(center, width).astype(np.int32)      # upper bound
    # Crop image
    image = image[lb[0]:ub[0],lb[1]:ub[1],lb[2]:ub[2]]
    return image



########################################
######### Training Dataloader ##########
########################################

class TrainingDataGenerator(DataLoader):
    """
    Dataloading class.
    Inspired by https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
    """
    def __init__(self,
                 data, 
                 batch_size,
                 raw_image_size,
                 n_channels=2,
                 hu_range=[-400,400],
                 num_threads_in_multithreaded=1,
                 seed_for_shuffle=1234,
                 return_incomplete=False, 
                 shuffle=True, 
                 infinite=True):
        super().__init__([row[1] for row in data.iterrows()], # DataLoader expects a list for the first argument 'data'
                         batch_size, 
                         num_threads_in_multithreaded,
                         seed_for_shuffle, 
                         return_incomplete, 
                         shuffle,
                         infinite)
        self.raw_image_size = raw_image_size
        self.n_channels     = n_channels
        self.hu_range       = hu_range
        self.indices        = list(range(len(self._data)))

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx                = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        
        # Initialize empty array for data
        data        = np.zeros((self.batch_size, self.n_channels, *self.raw_image_size), dtype=np.float32)
        labels      = np.empty(self.batch_size, dtype=np.int32)
        patient_ids = []
        
        # Iterate over patients_for_batch and include them in the batch
        for i, row in enumerate(patients_for_batch):
            
            # Get filenames
            ct_file  = row['preprocessed_ct_file']
            seg_file = row['preprocessed_segmentation_file']
            
            # Load data
            patient_data_ct   = np.zeros(shape=self.raw_image_size, dtype=np.float32)
            patient_data_seg  = np.zeros(shape=self.raw_image_size, dtype=np.float32)
            patient_data_ct   = load_ct(ct_file, hu_range=self.hu_range)
            
            # Add channel axis to combine CT image and segmentation
            patient_data_ct  = np.expand_dims(patient_data_ct, 0)
            
            # Prepare second channel
            if self.n_channels == 2:
   
                # Load segmentation
                patient_data_seg  = load_segmentation(seg_file, dataset=self.dataset)
                    
                # Add channel axis
                patient_data_seg = np.expand_dims(patient_data_seg, 0)
            
            patient_data = patient_data_ct
            if self.n_channels == 2:
                patient_data = np.concatenate((patient_data_ct, patient_data_blob))

            # Get label
            label = digitize_label(row['class_label'])
            
            # Fill data array
            data[i]   = patient_data
            labels[i] = (label)
            patient_ids.append(row['patient'])
        return {'data':data, 'label':labels, 'ids':patient_ids}
    


class BatchgenWrapper(tensorflow.keras.utils.Sequence):
    """
    Wrapper to make MIC-DKFZ batchgenerator work for Keras model.train() function.
    """
    def __init__(self,
                 data, 
                 batch_size,
                 raw_image_size,
                 patch_size,
                 n_channels=1,
                 non_linear=True,
                 one_hot=False,
                 num_processes=1,
                 num_cached_per_queue=1,
                 seeds=None,
                 pin_memory=False):
        self.dataloader    = TrainingDataGenerator(data=data,
                                                   batch_size=batch_size,
                                                   raw_image_size=raw_image_size,
                                                   n_channels=n_channels)
        self.transforms    = get_train_transforms(patch_size, non_linear=non_linear)
        self.datagenerator = MultiThreadedAugmenter(data_loader=self.dataloader,
                                                    transform=self.transforms,
                                                    num_processes=num_processes,
                                                    num_cached_per_queue=num_cached_per_queue,
                                                    seeds=seeds,
                                                    pin_memory=pin_memory)
        self.one_hot       = one_hot

    def __len__(self):
        return int(np.floor(len(self.dataloader._data) / self.dataloader.batch_size))
    
    def __getitem__(self, index):
        batch = next(self.datagenerator)
        # Input
        data  = batch['data']
        data  = np.moveaxis(data, 1, -1) # PyTorch format to Keras format
        label = batch['label']
        if self.one_hot:
            label = to_categorical(label)
        return data, label



##########################################
######### Validation Dataloader ##########
##########################################
    
class ValidationDataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 data,
                 batch_size,
                 patch_size,
                 n_channels=2,
                 hu_range=[-400,400],
                 num_threads=1,
                 shuffle=True):
        self.data        = data
        self.batch_size  = batch_size
        self.patch_size  = patch_size
        self.n_channels  = n_channels
        self.hu_range    = hu_range
        self.num_threads = num_threads
        self.shuffle     = shuffle
        self.indices     = list(range(self.data.shape[0]))

        # Multithreading
        self._lock = threading.Lock()
        
        # Initialize empty arrays/lists for batch data
        self.data_batch        = np.zeros((self.batch_size, self.n_channels, *self.patch_size), dtype=np.float32)
        self.labels_batch      = np.empty(self.batch_size, dtype=np.int32)
        self.patient_ids_batch = []
        
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        return self.generate_train_batch(index)
    
    def select_polyp_segs(self, index):
        return self.indices[index*self.batch_size:(index+1)*self.batch_size]

    def fill_batch(self, i, row):
        # Get filenames
        ct_file  = row['preprocessed_ct_file']
        seg_file = row['preprocessed_segmentation_file']
        
        # Load data
        patient_data_ct  = None
        patient_data_seg = None
        patient_data_ct  = load_ct(ct_file, hu_range=self.hu_range)
        
        # Crop image around the center to patch size
        patient_data_ct = crop_image(patient_data_ct, self.patch_size)

        # Add channel axis to combine CT image and segmentation
        patient_data_ct = np.expand_dims(patient_data_ct, -1)

        # Prepare second channel
        if self.n_channels == 2:
            # Load segmentation
            patient_data_seg  = load_segmentation(seg_file)

            # Crop image around the center to patch size
            patient_data_seg  = crop_image(patient_data_seg, self.patch_size)

            # Add channel axis
            patient_data_seg = np.expand_dims(patient_data_seg, -1)

        patient_data = patient_data_ct
        if self.n_channels == 2:
            patient_data = np.concatenate((patient_data_ct, patient_data_seg), axis=-1)
            
        # Get label
        label = digitize_label(row['class_label'])
        
        # Fill data array
        with self._lock:
            # Locked update
            self.data_batch[i]         = patient_data
            self.labels_batch[i]       = (label)
            self.patient_ids_batch.append(row['patient'])
    
    def generate_train_batch(self, index):
        # Select patietents for batch
        df_polyp_segs_batch = self.data.iloc[self.select_polyp_segs(index)]
        
        # Initialize empty array for data
        self.data_batch        = np.zeros((self.batch_size, *self.patch_size, self.n_channels), dtype=np.float32)
        self.labels_batch      = np.empty(self.batch_size, dtype=np.int32)
        self.patient_ids_batch = []
        
        # Iterate over patients_for_batch and include them in the batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i, row in enumerate(df_polyp_segs_batch.iterrows()):
                executor.submit(self.fill_batch, i, row[1]) # row = tuple, row[1] = DataFrame
        
        return self.data_batch, self.labels_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)