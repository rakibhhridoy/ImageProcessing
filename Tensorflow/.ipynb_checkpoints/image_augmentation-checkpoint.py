def image_augmentation(rescale_value,rotation_range,width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, fill_mode):
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    # rotation_range= 0-180 degree
    # height_shift_range = moving the subject inframe of the picture 0-100%
    # shear_range = offset vertically or horizontally 0-100%
    # fillmode = fills any pixels that had lost 'nearest', other

    train_datagen = ImageDataGenerator(rescale=1/rescale_value, rotation_range= rotation_range, 
        width_shift_range= width_shift_range, height_shift_range= height_shift_range, shear_range= shear_range,
        zoom_range= zoom_range, horizontal_flip= horizontal_flip, fill_mode= fill_mode)
    
    return train_datagen