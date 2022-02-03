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