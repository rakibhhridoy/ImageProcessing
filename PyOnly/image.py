import os
import zipfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class ImageProcessing:
    def __init__(self, base_dir = None, train_dirname = None, valid_dirname = None, 
                       class1_fn = None, class2_fn= None, zipped = False, 
                       zip_path = None, extracting_fn = None):
        
        '''
        If files are in zipped then give the path of the zipped file, ignore others

        -- base_dir: <string>      Base Directory/Folder path
        -- train_dirname: <string> Train Folder Name That contains Training Images
        -- valid_dir: <string>     Validation Folder Name That contains Validation Images
        -- class1_fn: <string>     Class 1 folder name 
        -- class2_fn: <string>     Class 2 folder name
        -- zipped: <Boolean>       If Images in zip file [Default False]
        -- zip_path: <string>      path of the zip file
        -- extracting_fn: <string> path of the folder to extract unzipped files
        
        Return The folders path including Images
        
        -- In Both Train and Validation Directory the class1 and class2   
        -- Folder name have to be same in train and validation folder
        
        Example:

            Train --> Cats --> [Image1.png, Image2.png, .......]
            Validation --> Cats --> [Image1.png, Image2.png, .......]
        
        Those nested Cats (Classification) Folder Name Should be same
        
        
        '''
        if zipped:
            
            zip_ref = zipfile.ZipFile(zip_path, 'r')

            zip_ref.extractall(extracting_fn)
            zip_ref.close()


            self.base_dir = zip_path
            self.train_dir_path = os.path.join(self.base_dir, train_dirname)
            self.validation_dir_path = os.path.join(self.base_dir, valid_dirname)

            # Directory with our training class1/class2 pictures
            self.train_class1_dir = os.path.join(self.train_dir_path, class1_fn)
            self.train_class2_dir = os.path.join(self.train_dir_path, class2_fn)

            # Directory with our validation class1/class2 pictures
            self.validation_class1_dir = os.path.join(self.validation_dir_path, class1_fn)
            self.validation_class2_dir = os.path.join(self.validation_dir_path, class2_fn)

            self.train_class1_fnames = os.listdir(self.train_class1_dir)
            self.train_class2_fnames = os.listdir(self.train_class2_dir)
            self.valid_class1_fnames = os.listdir(self.validation_class1_dir)
            self.valid_class2_fnames = os.listdir(self.validation_class2_dir)

        else:
            self.base_dir = base_dir
            self.train_dir_path = os.path.join(self.base_dir, train_dirname)
            self.validation_dir_path = os.path.join(self.base_dir, valid_dirname)

            # Directory with our training class1/class2 pictures
            self.train_class1_dir = os.path.join(self.train_dir_path, class1_fn)
            self.train_class2_dir = os.path.join(self.train_dir_path, class2_fn)

            # Directory with our validation class1/class2 pictures
            self.validation_class1_dir = os.path.join(self.validation_dir_path, class1_fn)
            self.validation_class2_dir = os.path.join(self.validation_dir_path, class2_fn)

            self.train_class1_fnames = os.listdir(self.train_class1_dir)
            self.train_class2_fnames = os.listdir(self.train_class2_dir)
            self.valid_class1_fnames = os.listdir(self.validation_class1_dir)
            self.valid_class2_fnames = os.listdir(self.validation_class2_dir)

    
    def __repr__(self):
        return "ImageProcessing"


    def paths(self):
        """
        <list>
        Returning all the paths of folders.

        [ Base Folder, Training Folder, Validation Folder, Train Class1 Folder,Train Class2 Folder , 
                                                Validation Class1 Folder, Validation Class2 Folder]

        """
        paths = [self.base_dir, self.train_dir_path, self.validation_dir_path, self.train_class1_dir, self.train_class2_dir, self.validation_class1_dir, self.validation_class2_dir]

        return paths

    def image_names(self, amount):

        """
        Showing the filename of the images on both Training And Validation..    
        
        
        -- amount: <int> Image Quantity to see
        Example: amount = 10
                    it will return 10 trainig class1 image name
                    it will return 10 trainig class2 image name
                    it will return 10 validation class1 image name
                    it will return 10 validation class2 image name
        """

        print(self.train_class1_fnames[:amount])
        print(self.train_class2_fnames[:amount])
        print(self.valid_class1_fnames[:amount])
        print(self.valid_class2_fnames[:amount])



#-- printing the amount of train and validation image
    def size(self, class1_name, class2_name):
        
        """
        Showing the number of images in train and validation folder..
        
        -- class1_name: <string> First Class Name
        -- class2_name: <string> Second Class Name

        """
        
        
        print(f'total training {class1_name} images :', len(os.listdir(self.train_class1_dir)))
        print(f'total training {class2_name} images :', len(os.listdir(self.train_class2_dir)))

        print(f'total validation {class1_name} images :', len(os.listdir(self.validation_class1_dir)))
        print(f'total validation {class2_name} images :', len(os.listdir(self.validation_class2_dir)))





# -- plotting the image

    def plot_image(self, amount):

        # Parameters for our graph; we'll output images in a (number_of_image / 2 * number_of_image / 2) configuration
        """
        Plotted Images of training images..

        -- amount: <int> Number of images to plot

        """



        nrows = amount / 2
        ncols = amount / 2

        pic_index = 0 
        
        # Index for iterating over images
        # Set up matplotlib fig, and size it to fit 4x4 pics


        fig = plt.gcf()
        fig.set_size_inches(ncols* amount / 2, nrows* amount / 2)

        pic_index+= amount

        next_class1_pix = [os.path.join(self.train_class1_dir, fname) 
                        for fname in self.train_class1_fnames[ pic_index- amount : pic_index] 
                        ]

        next_class2_pix = [os.path.join(self.train_class2_dir, fname) 
                        for fname in self.train_class2_fnames[ pic_index- amount :pic_index]
                        ]

        for i, img_path in enumerate(next_class1_pix+next_class2_pix):
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') # Don't show axes (or gridlines)

            img = mpimg.imread(img_path)
            plt.imshow(img)

        plt.show()



