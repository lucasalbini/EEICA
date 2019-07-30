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
import cv2
import matplotlib
matplotlib.use('agg') # esta linha e necessaria para usar a biblioteca sem uma interface grafica
import matplotlib.pyplot as plt

def read(name_img):
	img = cv2.imread(name_img)              # le a imagem
        img = cv2.resize(img, (96,96)).copy()   # faz resize para 96,96
        img = img / 128. - 1                    # normaliza entre -1 e 1
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
      )
                                                                                                   ####################### le apenas os primeiros mil dados de cada conjunto ###############     
file_x_train = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_train.npy') ####################### retirar [0:1000] para usar todos os dados #######################
y_train  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/y_train.npy')   ####################### retirar [0:1000] para usar todos os dados #######################
file_x_test  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits-species/X_test.npy')  ####################### retirar [0:1000] para usar todos os dados #######################

print 'reading train set...'
X_train = read_dataset(file_x_train)
y_train = to_categorical(y_train)
print 'reading test set...'
X_test  = read_dataset(file_x_test)

print 'shape of train set:', X_train.shape
print 'shape of test set:', X_test.shape

#datagen.fit(X_train, augment=True)

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

'''
#---LeNet5---

#Instantiate an empty model
model = Sequential()
# C1 Convolutional Layer
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(96,96,3), padding='same'))
# S2 Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
#DROPOUT
model.add(Dropout(0.25))
# C3 Convolutional Layer
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
# S4 Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#DROPOUT
model.add(Dropout(0.25))
# C5 Fully Connected Convolutional Layer
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(Flatten())
# FC6 Fully Connected Layer
model.add(Dense(84, activation='relu'))
#DROPOUT
model.add(Dropout(0.5))
#Output Layer with softmax activation
model.add(Dense(41, activation='softmax'))
'''
'''
#---AlexNet---

#Instantiate an empty model
model = Sequential()
# C1 Convolutional Layer
model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(96,96,3), padding='valid'))
# S2 Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
#DROPOUT
model.add(Dropout(0.25))
# C3 Convolutional Layer
model.add(Conv2D(256, kernel_size=(11, 11), strides=(1, 1), activation='relu', padding='valid'))
# S4 Pooling Layer
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#DROPOUT
model.add(Dropout(0.25))
#----
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid'))
#Flatten the CNN output so that we can connect it with fully connected layers
model.add(Flatten())
# FC6 Fully Connected Layer
model.add(Dense(84, activation='relu'))
#DROPOUT
model.add(Dropout(0.5))
#Output Layer with softmax activation
model.add(Dense(41, activation='softmax'))
'''

input_shape = (96, 96, 3)

#Instantiate an empty model
model = Sequential([

Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu', strides=(1, 1)),
#Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.25),

Conv2D(128, (3, 3), activation='relu', padding='same', strides=(1, 1)),
#Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.25),

Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1)),
#Conv2D(256, (3, 3), activation='relu', padding='same',),
#Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.25),

Conv2D(512, (3, 3), activation='relu', padding='same', strides=(1, 1)),
#Conv2D(512, (3, 3), activation='relu', padding='same',),
#Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Dropout(0.25),

#Conv2D(512, (3, 3), activation='relu', padding='same',),
#Conv2D(512, (3, 3), activation='relu', padding='same',),
#Conv2D(512, (3, 3), activation='relu', padding='same',),
#MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation='relu'),
Dropout(0.5),
#Dense(4096, activation='relu'),
Dense(41, activation='softmax')
])
model.summary()
#input()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = Adam()

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.25, batch_size=128, epochs=200, shuffle = True)
predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis = 1)

output_csv = open('/home/users/lucas/EEICA/Ex1/output_sample_species.csv', 'w')
for i,j in zip(file_x_test, predictions):
        output_csv.write(str(i.split('/')[-1]) + ';' + str(j) + '\n')
output_csv.close()

plot_losses(history, '/home/users/lucas/EEICA/Ex1/')

