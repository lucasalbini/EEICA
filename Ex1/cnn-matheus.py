import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read(name_img):
	img = cv2.imread(name_img)
	img = cv2.resize(img, (96,96)).copy()
	img = img / 128. - 1
	return img

def read_dataset(filelist):
	dataset = []
	img = None
	for x in xrange(0,filelist.shape[0]):
		img = read(filelist[x])
                #print x               
		dataset.append(img)
		del img

	return np.array(dataset)



file_x_train = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits/X_train.npy')
y_train  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits/y_train.npy')
file_x_test  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits/X_test.npy')
y_test  = np.load('/home/users/datasets/EEICA/UTFPR-BOP-splits/y_test.npy')

X_train = read_dataset(file_x_train)
X_test  = read_dataset(file_x_test)

y_train = to_categorical(y_train, num_classes = 6)
y_test = to_categorical(y_test, num_classes = 6)

print X_train.shape
print X_test.shape


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=25)
predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis = 1)
y_test = np.argmax(y_test, axis = 1)

from sklearn.metrics import classification_report, confusion_matrix
print classification_report(y_test, predictions)
print confusion_matrix(y_test, predictions)

output_csv = open('predictions_evaluation.csv', 'w')
for i,j in zip(file_x_test, predictions):
        output_csv.write(str(i.split('/')[-1]) + ';' + str(j) + '\n')
output_csv.close()



