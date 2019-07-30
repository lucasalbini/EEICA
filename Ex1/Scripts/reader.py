import keras
import tensorflow

from keras.preprocessing.image import ImageDataGenerator
# example of progressively loading images from file

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('/home/users/datasets/EEICA/UTFPR-BOP/', class_mode='categorical')