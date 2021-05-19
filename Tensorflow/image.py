                    ##--- Importing Different Module ---## 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from google.colab import files
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model







                    ###----- Labeled Image Resize || Augmentation || Both  -----### 

class ImageTensorProcessing:

    """
    Images should be labeled. If not labeled, then it can't predict accurately.. 
    This Class is combined form of doing boring steps in Tensorflow to predict classification problems..
    The Class has step by step processes as a functions.

    Steps:
        rescale()          --> rescale the image
        rescale_augment()  --> rescale the image using augmentation

    """

    def __init__(self, train_dir, validation_dir, class_type= None, target_size= None, batch_size= None):
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.class_type = class_type
        self.target_size = target_size
        self.batch_size = batch_size

        self.train_generator = None
        self.valid_generator = None
        self.train_aug_generator = None
        self.valid_aug_generator = None
        self.history = None

    
    def rescale(self, rescale):
        
        """
        Returning Scaled Image Generator Both Train And Validation Generator

        -- rescale: <int> [1, 255] rescaling by 1 / rescale
        
        Example:
            rescale = 255
            rescale will be 1 / 255
        
        
        Return train_generator, validation_generator
        """
        # All images will be rescaled by 1./255.
        train_datagen = ImageDataGenerator(rescale = 1.0/ rescale)
        test_datagen  = ImageDataGenerator(rescale = 1.0/ rescale)

        # --------------------
        # Flow training images in batches of 20 using train_datagen generator
        # --------------------
        self.train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            batch_size= self.batch_size,
                                                            class_mode=  self.class_type,
                                                            target_size= self.target_size)     
        # --------------------
        # Flow validation images in batches of 20 using test_datagen generator
        # --------------------
        self.valid_generator =  test_datagen.flow_from_directory(self.validation_dir,
                                                                batch_size= self.batch_size,
                                                                class_mode  =  self.class_type,
                                                                target_size = self.target_size)
        
        return self.train_generator , self.valid_generator



    
    def rescale_augment(self, rescale, rotation_range, width_shift_range, height_shift_range, 
                                        shear_range, zoom_range, horizontal_flip = True):
        
        """
        Returning Augmented Image Generator of Train and validation images..
        
        -- rescale: <int>          rescaling images in [1, 255]
        -- rotation_range: <float> is in degree
        -- width_shift_range:  <float>  shift the images in width
        -- height_shift_range: <float>  shift height or shift in relative of height
        -- shear_range: <float>         change in shear
        -- zoom_range: <float>          zooming level
        -- horizontal_flip: <Boolean>   Default True

        Return train_generator, validation_generator
        """
        
        train_datagen = ImageDataGenerator(
                                            rescale=1./ rescale,
                                            rotation_range= rotation_range,
                                            width_shift_range= width_shift_range,
                                            height_shift_range= height_shift_range,
                                            shear_range= shear_range,
                                            zoom_range= zoom_range,
                                            horizontal_flip= horizontal_flip,
                                            fill_mode='nearest')



        validation_datagen = ImageDataGenerator(rescale=1/ rescale)

        self.train_aug_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size= self.target_size,  
                                                            batch_size= self.batch_size,
                                                            class_mode= self.class_type)

        self.valid_aug_generator = validation_datagen.flow_from_directory(self.validation_dir,  
                                                                    target_size= self.target_size,
                                                                    batch_size= self.batch_size,
                                                                    class_mode= self.class_type)
        
        return self.train_aug_generator, self.valid_aug_generator
        
        

    
                            ###---- Callbacks ----###
    

    
# class my_callbacks(keras.callbacks.Callback):

#     def on_epoch_end(self, epoch, logd = {}, desire_loss):

#         if (logs.get('loss') < desire_loss):
#             self.model.stop_training = True

    
    

                            ###---- Training and Saving The Model at the same time ----###

        
# Checkpoints are the each epoch model data
    
    def train(self, modelname, model, epochs, aug = False):
        
        """
        Continue Training with a loop of saving model &
        Return history

        -- filepath: <str> Name the model to save
        -- model: <Object> ML / NeuralNetwork Model by Tensorflow
        -- epochs: <int> Number of Epochs
        -- aug: <int> generator to use.. Default False mean without augmented generator.


        """

        from keras.callbacks import ModelCheckpoint
        
        checkpoint = ModelCheckpoint(modelname, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]


        # fit the model
        if not aug:
            self.history = model.fit_generator(self.train_generator,validation_data= self.valid_generator, epochs= epochs, callbacks=callbacks_list)
        else:
            self.history = model.fit_generator(self.train_aug_generator,validation_data= self.valid_aug_generator, epochs= epochs, callbacks=callbacks_list)

        return self.history






                            ###---- Load The Saved Model & Fit The Model Again ----###



    # def load_fit_again_saved_model(self, x_train, filepath):
    #     # load the model
        
    #     new_model = load_model(filepath)
    #     assert_allclose(model.predict(x_train),
    #                     new_model.predict(x_train),
    #                     1e-5)

    #     # fit the model
    #     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    #     callbacks_list = [checkpoint]
    #     new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
            
        

    
    
    
    
    
    
                            ###---- Model Performance Evaluating ----###
            
            

    def evaluate_performance(self,modelname, model, epochs, aug = False):
        """
        Plotting the evaluation performance of the model

        run after running train()

        if you are running before train() then provide following:
            -- modelname: <str> name the model to be saved
            -- model: <object> provide custom model 
            -- epochs: <int> Number of Epochs
            -- aug: <int> generator to use.. Default False mean without augmented generator.

        """

        if self.history is not None:

            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

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


        else:

            self.train(modelname, model, epochs, aug)
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

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


    
    

    

    
    

                            ###---- Image Flow In Convolutional Layers ----###
    
    
    
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
  



