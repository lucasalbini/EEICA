import os
import numpy as np 
import keras
import cv2
import matplotlib
matplotlib.use('agg') # esta linha e necessaria para usar a biblioteca sem uma interface grafica
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
#Escolhe a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def schedule(epoch):
    if epoch < 500:
        return .01
    elif epoch < 600:
        return .002
    else:
        return .0004


def read(name_img):
	img = cv2.imread(name_img)              # le a imagem
        img = cv2.resize(img, (96,96)).copy()   # faz resize para 96,96
        #img = img / 128. - 1                    # normaliza entre -1 e 1
	return img

def read_dataset(filelist):
	dataset = []
        img = None
	for x in xrange(0,filelist.shape[0]):
                if x%1000 == 0:
                        print 'reading image', x, 'out of', filelist.shape[0]
		img = read(filelist[x])
		dataset.append(img)

        return np.array(dataset)

def plot_losses(history, output_path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(output_path + 'loss_plot.png', dpi = 300)
    plt.close()

#Leitura dos dados
#----------------------------------------------------------------------------------------------------------
file_x_train = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_train.npy')
y_train  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/y_train.npy')
file_x_test  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_test.npy')
#----------------------------------------------------------------------------------------------------------

print 'reading train set...'
X_train = read_dataset(file_x_train)
y_train = to_categorical(y_train)
print 'reading test set...'
X_test  = read_dataset(file_x_test)

print 'shape of train set:', X_train.shape
print 'shape of test set:', X_test.shape


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=[0.8,1],
        horizontal_flip=True,
        channel_shift_range = 30, 
        fill_mode='reflect',
        preprocessing_function = preprocess_input
      )

datagen_test = ImageDataGenerator(
        preprocessing_function = preprocess_input
      )

batch_size = 64

train_batches = datagen.flow_from_directory('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/train/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224), shuffle=True)
test_batches = datagen_test.flow_from_directory('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/test/', batch_size=batch_size, class_mode = 'categorical', target_size=(224, 224))

#vgg = applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224,224,3))
vgg = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,  input_shape=(224,224,3), pooling = 'avg')
model = Sequential()

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in vgg.layers:
        print layer.name
        #layer.trainable = False
vgg.summary()
model.add(vgg)
#model.add(layers.AveragePooling2D(pool_size=(5, 5)))
model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(41, activation='softmax'))


'''
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(41, activation='softmax'))
'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = keras.optimizers.Adam()

#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'top_k_categorical_accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy', 'top_k_categorical_accuracy'])

print model.summary()
lr_scheduler = LearningRateScheduler(schedule)
history =model.fit_generator(
		generator = train_batches,
		epochs=50,
		steps_per_epoch =  29731/batch_size,
		validation_steps = 5021/batch_size,
		validation_data = test_batches,
		use_multiprocessing=True,
		workers=8,
		callbacks=[lr_scheduler]
        )
#history = model.fit(X_train, y_train, validation_split=0.25, batch_size=64, epochs=15, shuffle = True)
#predictions = model.predict(X_test)
#predictions = model.predict(test_batches)
#predictions = np.argmax(predictions, axis = 1)

#output_csv = open('/home/users/lucas/EEICA/Ex1/output_sample_species.csv', 'w')
#for i,j in zip(file_x_test, predictions):
#        output_csv.write(str(i.split('/')[-1]) + ';' + str(j) + '\n')
#output_csv.close()

plot_losses(history, '/home/users/lucas/EEICA/Ex1/')



