# executar com python2 cnn.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # pode ser mudado para "1" se quiser usar a segunda gpu da maquina
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Activation, MaxPooling2D, AveragePooling2D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from keras.applications.inception_v3 import preprocess_input
from keras import applications
import cv2
import matplotlib
matplotlib.use('agg') # esta linha e necessaria para usar a biblioteca sem uma interface grafica
import matplotlib.pyplot as plt

#60 e 100
def schedule(epoch):
    if epoch < 500:
        return .1
    elif epoch < 600:
        return .005
    else:
        return .0025

def read(name_img):
	img = cv2.imread(name_img)              # le a imagem
        img = cv2.resize(img, (224,224)).copy()   # faz resize para 96,96
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

def read_testset(filelist):
	dataset = []
        img = None
	for x in xrange(0,filelist.shape[0]):
                if x%1000 == 0:
                        print 'reading image', x, 'out of', filelist.shape[0]
		img = read(filelist[x])
		img = img / 255.  
		dataset.append(img)

        return np.array(dataset)



def plot_losses(history, output_path):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(output_path + 'loss_plot-species.png', dpi = 300)
        plt.close()
                                                                                                   ####################### le apenas os primeiros mil dados de cada conjunto ###############     
file_x_train = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_train.npy') ####################### retirar [0:1000] para usar todos os dados #######################
y_train  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/y_train.npy')  ####################### retirar [0:1000] para usar todos os dados #######################


print 'reading train set...'
X_train = read_dataset(file_x_train)
y_train = to_categorical(y_train)


print 'shape of train set:', X_train.shape


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        #zoom_range=[0.8,1],
        horizontal_flip=True,
        #channel_shift_range = 30, 
        fill_mode='constant',
		cval=0,
        validation_split=0.25,
		preprocessing_function = preprocess_input
      )

testgen = ImageDataGenerator(
		rescale=1./255
	  )

train_generator = datagen.flow(
        X_train,
		y_train,
        batch_size=64,
        subset='training')

validation_generator = datagen.flow(
        X_train,
		y_train,
        batch_size=64,
        subset='validation')


#input_shape = (96, 96, 3)

#vgg = applications.VGG16(weights='imagenet', include_top=False,  input_shape=(224,224,3))
vgg = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,  input_shape=(224,224,3), pooling = 'avg')
model = Sequential()

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#for layer in vgg.layers:
#        print layer.name
        #layer.trainable = False
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

#input()
#kernel_regularizer=regularizers.l2(0.001)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = Adam()

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

lr_scheduler = LearningRateScheduler(schedule)

#history = model.fit_generator(train_generarator,validation_data= batch_size=128, epochs=140, shuffle = True, callbacks=[lr_scheduler])

history =model.fit_generator(
		generator = train_generator,
		epochs=200,
		steps_per_epoch =  29731/128,
		validation_steps = 7432/128,
		validation_data = validation_generator,
		shuffle=True,
		use_multiprocessing=True,
		workers=8,
		callbacks=[lr_scheduler]
        )

model.save_weights('weights-species.h5')

#-----------predict

file_x_test  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_test.npy')  ####################### retirar [0:1000] para usar todos os dados #######################
print 'reading test set...'
X_test  = read_testset(file_x_test)
print 'shape of test set:', X_test.shape
'''
test_generator = testgen.flow(
        X_test,
        batch_size=128,
        shuffle=False)  
'''

#predictions = model.predict_generator(test_generator, 41)
predictions = model.predict(X_test)


predictions = np.argmax(predictions, axis = 1)

output_csv = open('/home/users/lucas/EEICA/Ex2/output_sample_species.csv', 'w')
for i,j in zip(file_x_test, predictions):
        output_csv.write(str(i.split('/')[-1]) + ';' + str(j) + '\n')
output_csv.close()

plot_losses(history, '/home/users/lucas/EEICA/Ex2/')

