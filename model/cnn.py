from tensorflow.keras import layers, models



def identity_block(input_tensor,
                   filters,
                   kernel_size,
                   stage,
                   block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x



def conv_block(input_tensor,
               filters,
               kernel_size,
               stage,
               block,
               strides=2):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters, kernel_size, 
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters, kernel_size, 
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    shortcut = layers.Conv3D(filters, 1, 
                             strides=strides,
                             padding='same',
                             kernel_initializer='ones',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x



class MonteCarloDropout(layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

    

def CNN(weights=None,
        input_shape=None,
        classes=2,
        dropout=0.1,
        mc=False):

    img_input = layers.Input(shape=input_shape,
                             name='input')

    x = conv_block(img_input, filters=16, kernel_size=3, stage=1, block='a', strides=2)
    x = identity_block(x, filters=16, kernel_size=3, stage=1, block='b')

    x = conv_block(x, filters=32, kernel_size=3, stage=2, block='a', strides=2)
    x = identity_block(x, filters=32, kernel_size=3, stage=2, block='b')
    
    x = conv_block(x, filters=64, kernel_size=3, stage=3, block='a', strides=2)
    x = identity_block(x, filters=64, kernel_size=3, stage=3, block='b')

    x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
    
    # Add (Monte Carlo) Dropout
    if mc:
        x = MonteCarloDropout(dropout)(x)
    else:
        x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(classes, name='fc')(x)
    if classes==1:
        x = layers.Activation('sigmoid', name='sigmoid')(x)
    elif classes>1:
        x = layers.Activation('softmax', name='softmax')(x)

    # Create model.
    model = models.Model(img_input, x, name='cnn')

    # Load weights.
    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model