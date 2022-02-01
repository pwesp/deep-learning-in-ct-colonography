from   nrrd import read
import numpy as np
import SimpleITK as sitk



class Boxed_CT_Scan():
    """
    This class handles the preprocessing of a single polyp in a CT colonography 
    scan. First, the polyp is localized within the CTC scan using the polyp 
    segmentation mask. Afterwards, a small boxes around the polyp are extracted 
    from the scan and the segmetnation. These boxes are stored as 'boxed_ct' 
    and 'boxed_segmentation', respectively.

    Parameters
    ----------
    image : string
        Path to CT colonography scan
    segmentation : string
        Path to 3D segmentation mask labelling the polyp
    box_size : int (optional, default is 50)
        Size of subvolume around polyp volume center along each axis
    resample : boolean (optional, default is 'None')
        If 'True', the spacing of the CT scan is resampled to
        what is specified in 'new_scpacing'
    new_spacing : list of floats (optional, default is 'False')
        New spacing that should be used for resampling when 'resample'
        is set to 'True'
    roi_image : (optional, default is 'False')
        If 'True', a version of the boxed image is created where 
        only the segmentation is visible
    gaussian_blob : boolean (optional, default is 'False')
        If 'True', a box containing a 3D gaussian blob is created at 
        the center of the segmentation
    rescale : boolean (optional, default is 'False')
        Linearly rescales voxel values of boxed volumes, original and 
        augmented, into a defined range
    augmentation : boolean (optional, default is 'False')
        Controls if additional augmented versions of the volume will 
        be created

    Variables
    ---------
    boxed_ct : 3D array
        3D volume of small size cropped out of the ct around the polyp
    boxed_segmentation : 3D array
        3D volume of small size cropped out of the segmentation around the polyp
    """
    def __init__(self,
                 ct_path,
                 segmentation_path,
                 box_size       = 50,
                 resample       = False,
                 new_spacing    = [1.0,1.0,1.0],
                 rescale        = False):
        self.ct                 = self.__read_image(ct_path,           resample, new_spacing)
        self.segmentation       = self.__read_image(segmentation_path, resample, new_spacing, segmentation=True)
        self.size               = box_size
        self.rescale            = rescale
        self.segmented_voxels   = self.__localize_segmented_voxels(self.segmentation)
        self.center             = self.__localize_segmentation_center(self.segmented_voxels)
        self.boxed_ct           = self.__extract_boxed_volume(self.ct, self.center, self.size)
        self.boxed_segmentation = self.__extract_boxed_volume(self.segmentation, self.center, self.size)
        if self.rescale:
            self.__rescale()
    
    def __getitem__(self):
        return self.boxed_ct, self.segmentation
    
    def __read_image(self, filename, resample, new_spacing, segmentation=False):
        # In the original study, images were stored in a medical image format
        # However, in this example images are stored as numpy array instead and 
        # no meta information about the image is available
        image = np.load(filename)
        
        # For images stored in a medical image format, e.g. '.nii', '.nrrd' or '.dcm'
        # Load image and metadata from disk
        # image = sitk.ReadImage(filename)
        
        # Resample image
        if resample:
            image = sitk.GetImageFromArray(image) # Not required if image was stored as medical image file and loaded with SITK
            image = self.__resample_image(image, new_spacing=new_spacing, segmentation=segmentation)

        return image
    
    def __resample_image(self, image, new_spacing, segmentation=False):        
        # Set up resample image filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)
        if segmentation:
            resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputSpacing(tuple(new_spacing))

        # Calculate resampled image size
        orig_spacing = np.asarray(image.GetSpacing())
        new_spacing  = np.asarray(new_spacing)

        orig_size = np.array(image.GetSize(), dtype=np.int)
        factor    = np.divide(orig_spacing, new_spacing)
        new_size  = np.multiply(orig_size, factor)
        new_size  = np.rint(new_size).astype(np.int32)
        new_size  = new_size.astype(np.int32)
        new_size  = new_size.tolist()

        # Set new image size
        resampler.SetSize(new_size)

        # Resample image
        resampled_image = resampler.Execute(image)
        
        # Extract image as numpy array
        resampled_image = sitk.GetArrayFromImage(resampled_image)
        
        # Fix interpolated segmentation
        if segmentation:
            resampled_image[resampled_image>0.0] = 1.0  
        return resampled_image

    def __localize_segmented_voxels(self, segmentation):
        segmented_voxels = np.where(segmentation==1)
        return segmented_voxels
    
    def __localize_segmentation_center(self, segmented_voxels):
        # If no segmentation exists return image center
        if (segmented_voxels[0].shape[0]==0):
            return self.ct.shape[0], self.ct.shape[1], self.ct.shape[2]
        
        x_min = np.amin(segmented_voxels[0])
        x_max = np.amax(segmented_voxels[0])
        y_min = np.amin(segmented_voxels[1])
        y_max = np.amax(segmented_voxels[1])
        z_min = np.amin(segmented_voxels[2])
        z_max = np.amax(segmented_voxels[2])
        
        x_center = int((x_min + x_max) / 2.0)
        y_center = int((y_min + y_max) / 2.0)
        z_center = int((z_min + z_max) / 2.0)
        
        # Check if the boxed volume is completely inside the ct scan
        # If it is not, adjust the boxed volume accordingly by moving the center
        center = self.__check_image_bounds(x_center, y_center, z_center, self.size)
        return center
    
    def __extract_boxed_volume(self, volume, center, size):
        # Calculate polyp volume bounds inside the ct scan
        width  = int(size / 2.0)
        x_min  = center[0] - width
        y_min  = center[1] - width
        z_min  = center[2] - width
        if size % 2 == 1:
            width += 1
        x_max  = center[0] + width
        y_max  = center[1] + width
        z_max  = center[2] + width
        
        volume = volume[x_min:x_max,y_min:y_max,z_min:z_max]
        volume = volume.astype(np.int16)
        return volume
    
    def __check_image_bounds(self, x, y, z, size):
        centers = [x, y, z]
        width   = int(size / 2.0)
        
        # Check boxed volume bounds for each direction
        for i, center in enumerate(centers):
            # Correct center position if necessary
            if center < width:
                # Center too close to 0 voxel
                centers[i] += width-center
                # Center too close to end voxel
            elif center > (self.ct.shape[i]-width):
                centers[i] -= center-(self.ct.shape[i]-width)
        return centers[0], centers[1], centers[2]

    def __rescale(self):
        min_bound = -1024.0
        max_bound = 400.0
        new_bound = (max_bound - min_bound) / 2.0
        shift     = new_bound - max_bound
        
        self.boxed_ct = np.add(self.boxed_ct, shift)
        self.boxed_ct = np.divide(self.boxed_ct, new_bound)