import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def pre_process_image(image):
    image /= 127.5
    image -= 1.0
    return image


def pre_process_label(label):
    lbl = (label[:, :, 0]).astype(np.uint8)
    vehicle_indices = lbl == 10
    road_indices = lbl == 7
    roadline_indices = lbl == 6
    
    lbl = np.full_like(label, [1, 0, 0])
    r, c, ch = lbl.shape
    vehicle_indices[int(3*r/5):,:] = False
    lbl = np.full_like(label, [1, 0, 0])
    lbl[vehicle_indices] = [0, 1, 0]
    lbl[road_indices] = [0, 0, 1]
    lbl[roadline_indices] = [0, 0, 1]

    return lbl

def create_generators(image_dir='Train/CameraRGB',
                      label_dir='Train/CameraSeg',
                      batch_size=1,
                      target_size=(600, 800)):
    image_datagen = ImageDataGenerator(
        preprocessing_function=pre_process_image,
        #rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15)
    mask_datagen = ImageDataGenerator(
        preprocessing_function=pre_process_label,
        #rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15)
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        batch_size=batch_size,
        subset="training",
        interpolation="nearest",
        target_size=target_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        label_dir,
        class_mode=None,
        batch_size=batch_size,
        subset="training",
        interpolation="nearest",
        target_size=target_size,
        seed=seed)
    validate_image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        batch_size=batch_size,
        subset="validation",
        interpolation="nearest",
        target_size=target_size,
        seed=seed)

    validate_mask_generator = mask_datagen.flow_from_directory(
        label_dir,
        class_mode=None,
        batch_size=batch_size,
        subset="validation",
        interpolation="nearest",
        target_size=target_size,
        seed=seed)

    # Provide the same seed and keyword arguments to the fit and flow methods
    #image_datagen.fit(images, seed=seed)
    #mask_datagen.fit(masks, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    validate_generator = zip(validate_image_generator, validate_mask_generator)

    return (train_generator, validate_generator)

