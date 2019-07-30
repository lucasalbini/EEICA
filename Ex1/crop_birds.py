import /home/users/lucas/DataMining/Yolo/darknet/darknet
import os, sys
import numpy as np
from PIL import Image
import glob
from glob import glob
import cv2

folder = "/home/users/datasets/EEICA/UTFPR-BOP-splits-species/train/"

def crop(folder):
	#Carrega a rede neural e os pesos
	net = darknet.load_net("/home/users/lucas/DataMining/Yolo/darknet/cfg/yolov3.cfg", "/home/users/lucas/DataMining/Yolo/darknet/yolov3.weights", 0)
	meta = darknet.load_meta("/home/users/lucas/DataMining/Yolo/darknet/cfg/coco.data")
	
	specie = os.listdir(folder)
	for bird in sorted(specie):
		path = bytes(os.path.join(folder, bird).encode("utf-8"))
		res = darknet.detect(net, meta, path)
		#res[:1] mostra objeto localizado // (res[:1])[0] mostra posicao na lista do objeto localizado(NAO ALTERAR) // (res[:])[0][1] mostra o elemento dentro
		#print ((res[:])[0][0])	
		v = []	
		for k in range(len((res[:]))):
			name = ((res[:])[k][0])
			#print name
			acc = ((res[:])[k][1])
			print (str(f)+ " - " + str(name) + " - " + str(acc))

			x = ((res[:])[k][2][0])
			y = ((res[:])[k][2][1])
			w = ((res[:])[k][2][2])
			h = ((res[:])[k][2][3])

			area = w*h
			v.append(area)


		if (len(v)>0):
			ind_max=np.argmax(v)

			x1 = ((res[:])[ind_max][2][0])
			y1 = ((res[:])[ind_max][2][1])
			w1 = ((res[:])[ind_max][2][2])
			h1 = ((res[:])[ind_max][2][3])

			x_max = (2*x1+w1)/2
			x_min = (2*x1-w1)/2
			y_min = (2*y1-h1)/2
			y_max = (2*y1+h1)/2
			image = Image.open(path2)
			cropped = image.crop((x_min, y_min, x_max, y_max))

			try:
				os.makedirs('/home/users/lucas/EEICA/Ex1/Data/')
			except Exception as e:
		 		pass    	
			saving_path = "/home/users/lucas/EEICA/Ex1/Data/"+str(bird)
			print "cropped"
			save_file = open(saving_path, 'w')
			cropped.save(saving_path)
			save_file.close()
		

crop(folder)







































'''
import cv2
import sys
import numpy as np




def load_image(path):
        img = cv2.imread(path)
        return img

def load_label(path):
        label = ''
        lookup = path.split('/')[0].lower()
        with open('meta/classes.txt') as myFile:
            for num, line in enumerate(myFile, 1):
                if lookup in line.lower():
                    label = num
        return label-1


def preprocess_input(image):
    return 0



def image_generator(files,label_file, batch_size = 64):
    
    while True:

          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = files, 
                                         size = batch_size)
          batch_input = []
          batch_output = [] 
          
          # Read in each input, perform preprocessing and get labels

          for input_path in batch_paths:

              input = get_input(input_path )
              output = get_output(input_path,label_file=label_file )
            
              input = preprocess_input(image=input)
              batch_input += [ input ]
              batch_output += [ output ]

          # Return a tuple of (input,output) to feed the network

          batch_x = np.array( batch_input )
          batch_y = np.array( batch_output )
        
          yield( batch_x, batch_y )




def load_dataset(paths_file):
        # declara listas vazias para armazenar os dados
        data = []
        labels = []
        label_names = []

        # abre o arquivo que especifica quais imagens devem ser carregadas
        with open(paths_file) as f:
                lines = f.readlines()
        lines = [x.strip() for x in lines] 

        # debugging purposes
        #perm = np.random.permutation(len(lines))[0:500]
        #print perm
        #lines = np.array(lines)[perm] 

        # percorre a lista de imagens a serem carregadas e armazena as imagens e labels em suas respectivas listas
        for i,j in enumerate(lines):
                sys.stdout.write("Reading Images: %d%%   \r" % (i/float(len(lines))*100.))
                sys.stdout.flush()
                image = cv2.imread('images/'+str(j)+'.jpg')
                data.append(image)
                label_names.append(j.split('/')[0])

        # converte os nomes das classes para numeros
        class_names = sorted(list(set(label_names)))
        for l in label_names:
                labels.append(np.uint8(class_names.index(l)))
        
        return data, labels, class_names

#data , labels, class_names = load_dataset('meta/train.txt')
#print labels, len(labels)



print load_label('baby_back_ribs/3691980')
'''              

