import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
files = []
number = np.zeros((6))
from shutil import copyfile

files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Accipitridae/*.jpg'))
files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Cathartidae/*.jpg'))
files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Falconidae/*.jpg'))
files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Pandionidae/*.jpg'))
files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Strigidae/*.jpg'))
files.append(glob.glob('/home/users/datasets/EEICA/UTFPR-BOP/Tytonidae/*.jpg'))

count = 0 
for i in range(0,6):
	count = count + len(files[i])
	number[i] = len(files[i])

x = []
y = []

for i in range(0,6):
	for j in range(0,int(number[i])):
		y.append(i)
		x.append(files[i][j])

X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size = 0.3, train_size=0.7, random_state = 42, shuffle = True, stratify = y)

new_X_train = []
new_X_test = []
print len(X_train), len(X_test), len(y_train), len(y_test)

for d in X_train:
        print 'copying ',d
        try:
                os.makedirs("/home/users/datasets/EEICA/UTFPR-BOP-splits/train/"+d.split('/')[6])
        except:
                print 'operacao concluida com sucesso'

        copyfile(d, "/home/users/datasets/EEICA/UTFPR-BOP-splits/train/"+d.split('/')[6] + '/' + d.split('/')[7])
        new_X_train.append("/home/users/datasets/EEICA/UTFPR-BOP-splits/train/"+d.split('/')[6] + '/' + d.split('/')[7])


for d in X_test:
        print 'copying ',d
        try:
                os.makedirs("/home/users/datasets/EEICA/UTFPR-BOP-splits/test/")
        except:
                print 'operacao concluida com sucesso'

        copyfile(d, "/home/users/datasets/EEICA/UTFPR-BOP-splits/test/" + d.split('/')[7])
        new_X_test.append("/home/users/datasets/EEICA/UTFPR-BOP-splits/test/" + d.split('/')[7])


new_X_train = np.array(new_X_train)
new_X_test = np.array(new_X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

np.save('X_train.npy', new_X_train)
np.save('X_test.npy', new_X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)








