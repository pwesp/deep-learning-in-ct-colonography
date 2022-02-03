import concurrent.futures
import numpy as np
from   pathlib import Path
import tensorflow.keras
import threading



################################################
######### Shared Dataloader Functions ##########
################################################

def digitize_label(label_string):
    if label_string=='benign' or label_string=='001':
        return 0
    elif label_string=='precancerous' or label_string=='1':
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
        self.indices     = list(range(len(data)))

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
    
    def select_patients(self, index):
        return self.indices[index*self.batch_size:(index+1)*self.batch_size]

    def fill_batch(self, i, j):
        # Get filenames
        patient_files = [str(x.absolute()) for x in Path(j).iterdir() if x.is_file()]
        patient_files.sort()
        
        # Load data
        patient_data_ct  = None
        patient_data_seg = None
        patient_data_ct  = load_ct(patient_files[0], hu_range=self.hu_range)
        
        # Crop image around the center to patch size
        patient_data_ct = crop_image(patient_data_ct, self.patch_size)

        # Add channel axis to combine CT image and segmentation
        patient_data_ct = np.expand_dims(patient_data_ct, -1)

        # Prepare second channel
        if self.n_channels == 2:
            # Load segmentation
            patient_data_seg  = load_segmentation(patient_files[1])

            # Crop image around the center to patch size
            patient_data_seg  = crop_image(patient_data_seg, self.patch_size)

            # Add channel axis
            patient_data_seg = np.expand_dims(patient_data_seg, -1)

        patient_data = patient_data_ct
        if self.n_channels == 2:
            patient_data = np.concatenate((patient_data_ct, patient_data_seg), axis=-1)

        # Get label
        label = digitize_label(j.split('/')[-1])
        
        # Fill data array
        with self._lock:
            # Locked update
            self.data_batch[i]         = patient_data
            self.labels_batch[i]       = (label)
            self.patient_ids_batch.append(j.split('/')[-2])
    
    def generate_train_batch(self, index):
        # Select patietents for batch
        patients_for_batch = [self.data[patient] for patient in self.select_patients(index)]
        
        # Initialize empty array for data
        self.data_batch        = np.zeros((self.batch_size, *self.patch_size, self.n_channels), dtype=np.float32)
        self.labels_batch      = np.empty(self.batch_size, dtype=np.int32)
        self.patient_ids_batch = []
        
        # Iterate over patients_for_batch and include them in the batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i, j in enumerate(patients_for_batch):
                executor.submit(self.fill_batch, i, j)
        
        return self.data_batch, self.labels_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)