from   batchgenerators.transforms import Compose
from   batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from   batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from   batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
import numpy as np



def get_train_transforms(patch_size, non_linear=True):
    """
    Online data augmentations for 3D data.
    
    Inspired by https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
    """
    # List of transforms
    train_transforms = []

    if non_linear:  
        # Spatial transform with deformation
        train_transforms.append(
            SpatialTransform_2(
                patch_size, [i // 1.25 for i in patch_size],
                do_elastic_deform=True, deformation_scale=(0, 0.25),
                do_rotation=True,
                angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                do_scale=True, scale=(0.75, 1.25),
                border_mode_data='constant', border_cval_data=0,
                border_mode_seg='constant', border_cval_seg=0,
                order_seg=1, order_data=3,
                random_crop=True,
                p_el_per_sample=0.15, p_rot_per_sample=0.15, p_scale_per_sample=0.15
            )
        )

        # Gamma transform (nonlinear transformation of intensity values, https://en.wikipedia.org/wiki/Gamma_correction)
        train_transforms.append(GammaTransform(gamma_range=(0.75, 1.25), invert_image=False, per_channel=False, p_per_sample=0.15))
    else:
        # Spatial transform
        train_transforms.append(
            SpatialTransform_2(
                patch_size, [i // 1.25 for i in patch_size],
                do_elastic_deform=False, deformation_scale=(0, 0),
                do_rotation=True,
                angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                do_scale=True, scale=(0.75, 1.25),
                border_mode_data='constant', border_cval_data=0,
                border_mode_seg='constant', border_cval_seg=0,
                order_seg=1, order_data=3,
                random_crop=True,
                p_el_per_sample=0.0, p_rot_per_sample=0.15, p_scale_per_sample=0.15
            )
        )
    
    # Mirror transform along all axes
    train_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # Brightness transform for 15% of samples
    train_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), per_channel=False, p_per_sample=0.15))

    # Add Gaussian Noise for 15% of samples
    train_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # Add Gaussian Blurring for 15% of samples
    train_transforms.append(GaussianBlurTransform(blur_sigma=(0.75, 1.25), different_sigma_per_channel=False, p_per_channel=1.0, p_per_sample=0.15))
    
    # Compose all transforms together
    train_transforms = Compose(train_transforms)
    
    return train_transforms