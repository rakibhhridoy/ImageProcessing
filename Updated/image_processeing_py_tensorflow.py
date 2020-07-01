                                
                                ####---- Python Part ----####




import os
import zipfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt




                        #**** If File is Zipped ****#




def unzipping_file(path, before_path_fn):
    local_zip = path

    zip_ref = zipfile.ZipFile(local_zip, 'r')

    zip_ref.extractall(before_path_fn)
    zip_ref.close()









                    #####---- Joining The Directory to each other ----####


def joining_directory_to_each_other_binary_class(base_directory_path, train_fn, validation_fn, class1_fn, class2_fn):
    
  # -- In Both Train and Validation Directory the class1 and class2   --Folder name have to be same--

  base_dir = base_directory_path

  train_dir = os.path.join(base_dir, train_fn)
  validation_dir = os.path.join(base_dir, validation_fn)

  # Directory with our training class1/class2 pictures
  train_class1_dir = os.path.join(train_dir, class1_fn)
  train_class2_dir = os.path.join(train_dir, class2_fn)

  # Directory with our validation class1/class2 pictures
  validation_class1_dir = os.path.join(validation_dir, class1_fn)
  validation_class2_dir = os.path.join(validation_dir, class2_fn)

  return train_dir, validation_dir, train_class1_dir , train_class2_dir, validation_class1_dir, validation_class2_dir









                    ##--- Image Filename Showing --##


def training_filename_show(train_class1_dir, train_class2_dir, amount):
    
  # -- amount is for how many image name do we want to see

  train_class1_fnames = os.listdir( train_class1_dir)
  train_class2_fnames = os.listdir( train_class2_dir)

  print(train_class1_fnames[:amount])
  print(train_class2_fnames[:amount])

  return train_class1_fnames, train_class2_fnames



                    #-- printing the amount of train and validation image --#


def train_validation_size(train_class1_dir, train_class2_dir, validation_class1_dir, validation_class2_dir, class1_name, class2_name):
    
  print(f'total training {class1_name} images :', len(os.listdir(      train_class1_dir ) ))
  print(f'total training {class2_name} images :', len(os.listdir(      train_class2_dir ) ))

  print(f'total validation {class1_name} images :', len(os.listdir( validation_class1_dir ) ))
  print(f'total validation {class2_name} images :', len(os.listdir( validation_class2_dir ) ))





                    #-- plotting the image --#


def plotting_image(number_of_image, train_class1_fnames, train_class2_fnames):

    # Parameters for our graph; we'll output images in a (number_of_image / 2 * number_of_image / 2) configuration

    nrows = number_of_image / 2
    ncols = number_of_image / 2

    pic_index = 0 
    
    # Index for iterating over images
    # Set up matplotlib fig, and size it to fit 4x4 pics


    fig = plt.gcf()
    fig.set_size_inches(ncols* number_of_image / 2, nrows* number_of_image / 2)

    pic_index+= number_of_image

    next_class1_pix = [os.path.join(train_class1_dir, fname) 
                    for fname in train_class1_fnames[ pic_index- number_of_image : pic_index] 
                    ]

    next_class2_pix = [os.path.join(train_class2_dir, fname) 
                    for fname in train_class2_fnames[ pic_index- number_of_image :pic_index]
                    ]

    for i, img_path in enumerate(next_class1_pix+next_class2_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()











                    ####---- TensorFlow Part ----####



                  ##--- Importing Different Module ---## 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from google.colab import files
import matplotlib.pyplot as plt
from keras.preprocessing import image








                    ###----- Labeled Image Resize || Augmentation || Both  -----### 




def resize_labeled_images(train_dir, validation_dir, rescale, class_type, target_size, batch_size):
    
    

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator( rescale = 1.0/ rescale )
    test_datagen  = ImageDataGenerator( rescale = 1.0/ rescale)

    # --------------------
    # Flow training images in batches of 20 using train_datagen generator
    # --------------------
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size= batch_size,
                                                        class_mode=  class_type,
                                                        target_size= target_size)     
    # --------------------
    # Flow validation images in batches of 20 using test_datagen generator
    # --------------------
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                            batch_size= batch_size,
                                                            class_mode  =  class_type,
                                                            target_size = target_size)
    
    return train_generator , validation_generator



    
def resize_labeled_images_augmentation(train_dir, validation_dir, target_size, batch_size, class_type,rescale, rotation_range,width_shift_range,height_shift_range, 
                                       shear_range, zoom_range):
    
    
    train_datagen = ImageDataGenerator(
                                        rescale=1./ rescale,
                                        rotation_range= rotation_range,
                                        width_shift_range= width_shift_range,
                                        height_shift_range= height_shift_range,
                                        shear_range= shear_range,
                                        zoom_range= zoom_range,
                                        horizontal_flip= True,
                                        fill_mode='nearest')



    validation_datagen = ImageDataGenerator(rescale=1/ rescale)

    train_generator_augmented = train_datagen.flow_from_directory(train_dir,
                                                        target_size= target_size,  
                                                        batch_size= batch_size,
                                                        class_mode= class_type)

    validation_generator_augmented = validation_datagen.flow_from_directory(validation_dir,  
                                                                target_size= target_size,
                                                                batch_size= batch_size,
                                                                class_mode= class_type)
    
    return train_generator_augmented , validation_generator_augmented
    
    


    
                                        ###---- Callbacks ----###
    

    
class my_callbacks(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logd = {}, desire_loss):

        if (logs.get('loss') < desire_loss):
            self.model.stop_training = True

    
    




                        ###---- Training and Saving The Model at the same time ----###

        
# Checkpoints are the each epoch model data
    
def training_with_loading(filepath, model):
    
    from keras.callbacks import ModelCheckpoint
    
    filepath = 'model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


    # fit the model
    history = model.fit_generator(train_generator,validation_data= validation_generator, epochs= 20, callbacks=callbacks_list)
    
    return history






                            ###---- Load The Saved Model & Fit The Model Again ----###



def load_fit_again_saved_model(x_train, filepath):
    # load the model
    
    new_model = load_model(filepath)
    assert_allclose(model.predict(x_train),
                    new_model.predict(x_train),
                    1e-5)

    # fit the model
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
        
    

    
    
    
    
    
    
                            ###---- Model Performance Evaluating ----###
            
            

def evaluate_the_model_performance(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


    
    

    

    
    

                            ###---- Image Flow In Conolutional Layers ----###
    
    
    
def images_in_conv_layer(rescale, train_class1_dir, train_class2_dir, train_class1_fnames, train_class2_fnames):
    

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

    # Let's prepare a random input image of a cat or dog from the training set.
    class1_img_files = [os.path.join(train_class1_dir, f) for f in train_class1_fnames]
    class2_img_files = [os.path.join(train_class2_dir, f) for f in train_class2_fnames]

    img_path = random.choice(class1_img_files + class2_img_files)
    img = load_img(img_path, target_size= target_size)  # this is a PIL image

    x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
    x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= rescale

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        
        if len(feature_map.shape) == 4:
        
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
            
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            
            #-------------------------------------------------
            # Postprocess the feature to be visually palatable
            #-------------------------------------------------
            for i in range(n_features):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std ()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

            #-----------------
            # Display the grid
            #-----------------

            scale = 20. / n_features
            plt.figure( figsize=(scale * n_features, scale) )
            plt.title ( layer_name )
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 



            
            
            
            
            
            
            
                        ###---- Test Image ---###
                
                
                
                    ###---- This is for Colab ---###
                    
                    
                    
# Uploading any image to test


def upload_image_to_test(class1_name, class2_name, directory_on_working, target_size):
    
    

    uploaded=files.upload()

    for fn in uploaded.keys():
    
        # predicting images
        path= directory_on_working + fn
        img=image.load_img(path, target_size= target_size)
        
        x=image.img_to_array(img)
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        classes = model.predict(images, batch_size=10)
        
        print(classes[0])
        
        if classes[0]>0:
            print(fn + f" is a {class1_name}")  
        
        else:
            print(fn + f" is a {class2_name}")  

            
            


                        ###---- Manual ----###

                
def test_image(path,class1_name, class2_name):
    
    
    # predicting images
    path= directory_on_working + fn
    img=image.load_img(path, target_size= target_size)

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)

    print(classes[0])

    if classes[0]>0:
        print(fn + f" is a {class1_name}")  

    else:
        print(fn + f" is a {class2_name}")




def clean_up():
    import os, signal

    os.kill(   os.getpid() , 
          signal.SIGKILL
        )
  



