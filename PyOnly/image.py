import os
import zipfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def unzipping_file(path, before_path_fn):
  local_zip = path

  zip_ref = zipfile.ZipFile(local_zip, 'r')

  zip_ref.extractall(before_path_fn)
  zip_ref.close()


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


def training_filename_show(train_class1_dir, train_class2_dir, amount):
    
  # -- amount is for how many image name do we want to see

  train_class1_fnames = os.listdir( train_class1_dir)
  train_class2_fnames = os.listdir( train_class2_dir)

  print(train_class1_fnames[:amount])
  print(train_class2_fnames[:amount])

  return train_class1_fnames, train_class2_fnames



#-- printing the amount of train and validation image
def train_validation_size(train_class1_dir, train_class2_dir, validation_class1_dir, validation_class2_dir, class1_name, class2_name):
    
  print(f'total training {class1_name} images :', len(os.listdir(      train_class1_dir ) ))
  print(f'total training {class2_name} images :', len(os.listdir(      train_class2_dir ) ))

  print(f'total validation {class1_name} images :', len(os.listdir( validation_class1_dir ) ))
  print(f'total validation {class2_name} images :', len(os.listdir( validation_class2_dir ) ))





# -- plotting the image

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



