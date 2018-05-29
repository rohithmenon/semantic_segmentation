from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Softmax, UpSampling2D, ZeroPadding2D, concatenate


def vgg16(dropout=0.5, target_size=(600, 800)):
    vgg16 = VGG16(include_top=False, input_shape=(target_size[0], target_size[1], 3))
    for layer in vgg16.layers:
        layer.trainable = False

    block3_pool = vgg16.get_layer('block3_pool')
    block4_pool = vgg16.get_layer('block4_pool')
    block5_pool = vgg16.get_layer('block5_pool')

    x = Conv2D(filters=4096, kernel_size=7, padding="same", activation="relu", name="fc6")(block5_pool.output)
    x = BatchNormalization()(x)
    #x = Dropout(dropout, name="fc6_dropout")(x)
    x = Conv2D(filters=4096, kernel_size=1, padding="same", activation="relu", name="fc7")(x)
    x = BatchNormalization()(x)
    #x = Dropout(dropout, name="fc7_dropout")(x)
    x = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="fcn_32")(x)
    #x = UpSampling2D(size=2, name="fcn32_2x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="same", name="fcn32_2x")(x)
    #x = ZeroPadding2D(padding=(1, 0), name="fcn32_2x_pad")(x)
    #x = Cropping2D(cropping=((0, 1), (0, 0)), name="fcn32_2x_crop")(x)

    block4_pred = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="block_4_pred")(block4_pool.output)
    x = concatenate([x, block4_pred], name="fcn_16")
    
    #x = UpSampling2D(size=2, name="fcn16_2x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2), padding="same", name="fcn16_2x")(x)
    #x = ZeroPadding2D(padding=(0, 1), name="fcn16_2x_pad")(x)
    #x = Cropping2D(cropping=((0, 0), (0, 1)), name="fcn16_2x_crop")(x)
    block3_pred = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="block_3_pred")(block3_pool.output)
    x = concatenate([x, block3_pred], name="fcn_8")

    x = Conv2D(filters=3, kernel_size=1, padding="same", activation="relu", name="prediction")(x)
    #x = UpSampling2D(size=8, name="pred_8x")(x)
    x = Conv2DTranspose(filters=3, kernel_size=16, strides=(8, 8), padding="same", name="pred_8x")(x)
    x = ZeroPadding2D(padding=(2, 2), name="pred_8x_pad")(x)
    x = Cropping2D(cropping=((0, 1), (0, 1)), name="fcn8_2x_crop")(x)
    x = Softmax(name="pred_softmax")(x)

    return Model(vgg16.input, x, name="sem_seg")
