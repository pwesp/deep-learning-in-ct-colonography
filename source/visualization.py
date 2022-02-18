from   matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np



def plotter_batch(batch, sample_nr, channel, slice_x, slice_y, slice_z, cmap, reverse_cmap):
    image = batch[0]
    label = batch[1]
    
    # Colormap
    color_map = plt.cm.get_cmap(cmap)
    if reverse_cmap:
        color_map = color_map.reversed()
    
    print('CT Image     {:d}: {} [{:.2f}, {:.2f}]'.format(sample_nr, image[sample_nr,:,:,:,0].shape, np.amin(image[sample_nr,:,:,:,0]), np.amax(image[sample_nr,:,:,:,0])))
    if channel==2:
        print('2nd channel  {:d}: {} [{:.2f}, {:.2f}]'.format(sample_nr, image[sample_nr,:,:,:,1].shape, np.amin(image[sample_nr,:,:,:,1]), np.amax(image[sample_nr,:,:,:,1])))
    print('Label        {}'.format(label[sample_nr]))
    
    fig, ax = plt.subplots(1, 3, figsize=(21,7))
    ax[0].imshow(image[sample_nr,slice_x,:,:,channel], cmap=color_map)
    ax[1].imshow(image[sample_nr,:,slice_y,:,channel], cmap=color_map)
    ax[2].imshow(image[sample_nr,:,:,slice_z,channel], cmap=color_map)
    plt.show()
    
    

def plotter_gradcam(X, cam, seg, cam_nr, modality, y_true, y_pred, sample_nr, slice_x, slice_y, slice_z, alpha, threshold, cmap, reverse_cmap):
     
    # Pick and normalize image
    X = np.copy(X[sample_nr])
    X[...,0] = np.divide(X[...,0],np.amax(X[...,0]))
    if modality>0:
        X[...,1] = np.divide(X[...,1],np.amax(X[...,1]))
    
    # Pick cam
    cam = np.copy(cam[cam_nr,sample_nr,...])
    
    # Pick segmentation
    seg = np.copy(seg[sample_nr])
    
    # Colormap
    color_map = plt.cm.get_cmap(cmap)
    if reverse_cmap:
        color_map = color_map.reversed()
        
    print('Image {:d}: [{:.2f}, {:.2f}]'.format(sample_nr,np.amin(X),np.amax(X)))
    print('Cam   {:d}: [{:.2f}, {:.2f}]'.format(sample_nr,np.amin(cam),np.amax(cam)))
    print('True label: {:d}'.format(y_true[sample_nr]))
    print('Prediction: {:.2f}'.format(y_pred[sample_nr]))
    
    # Threshold
    X_in = np.zeros_like(X[...,modality])
    X_in[np.where(cam>=threshold)]=1
    X_out = np.zeros_like(X[...,modality])
    
    # Figure
    fig, ax = plt.subplots(2, 3, figsize=(14,7.6), sharex=True, sharey=True)
    
    # Original input image with GradCAM on top
    ax[0,0].imshow(X[slice_x,:,:,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())
    ax[0,1].imshow(X[:,slice_y,:,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())
    ax[0,2].imshow(X[:,:,slice_z,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())

    ax[0,0].imshow(cam[slice_x,:,:], cmap=color_map, vmin=0.0, vmax=1.0, alpha=alpha) # overlay
    ax[0,1].imshow(cam[:,slice_y,:], cmap=color_map, vmin=0.0, vmax=1.0, alpha=alpha) # overlay
    ax[0,2].imshow(cam[:,:,slice_z], cmap=color_map, vmin=0.0, vmax=1.0, alpha=alpha) # overlay
    
    ax[0,0].set_xlabel('y', fontsize=16)
    ax[0,0].set_ylabel('z', fontsize=16)
    ax[0,1].set_xlabel('x', fontsize=16)
    ax[0,1].set_ylabel('z', fontsize=16)
    ax[0,2].set_xlabel('x', fontsize=16)
    ax[0,2].set_ylabel('y', fontsize=16)
    
    im = plt.imshow(np.zeros((50,50)), cmap=color_map, vmin=0.0, vmax=1.0, alpha=1.0)
    
    # Binarized CT
    ax[1,0].imshow(X[slice_x,:,:,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())
    ax[1,1].imshow(X[:,slice_y,:,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())
    ax[1,2].imshow(X[:,:,slice_z,modality], cmap=plt.cm.get_cmap('gist_yarg').reversed())
    
    ax[1,0].imshow(X_in[slice_x,:,:], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    ax[1,1].imshow(X_in[:,slice_y,:], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    ax[1,2].imshow(X_in[:,:,slice_z], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    
    ax[1,0].imshow(X_out[slice_x,:,:], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    ax[1,1].imshow(X_out[:,slice_y,:], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    ax[1,2].imshow(X_out[:,:,slice_z], cmap=plt.cm.get_cmap('bwr'), alpha=0.6)
    
    ax[1,0].set_xlabel('y', fontsize=16)
    ax[1,0].set_ylabel('z', fontsize=16)
    ax[1,1].set_xlabel('x', fontsize=16)
    ax[1,1].set_ylabel('z', fontsize=16)
    ax[1,2].set_xlabel('x', fontsize=16)
    ax[1,2].set_ylabel('y', fontsize=16)
    
    # Settings
    for axis in ax.flatten():
        axis.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.8, 0.09, 0.025, 0.885])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=14)
    
    plt.show()