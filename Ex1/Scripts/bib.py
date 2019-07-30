import cv2
import numpy as np
import os
from glob import glob
import itertools
from random import randint
from keras.utils.np_utils import to_categorical
import time 
import keras

PATH_TO_FRAMES = '/home/users/datasets/UCF-101_opticalflow/'
#PATH_TO_FRAMES = '/home/ssd/ucf101/'
PATH_TO_TRAIN_TEST_SPLITS = '/home/users/matheus/doutorado/i3d/keras-kinetics-i3d/data/ucf101/'
TRAIN_TEST_SPLITS_FILENAMES = [['trainlist01.txt', 'testlist01.txt'], ['trainlist02.txt', 'testlist02.txt'], ['trainlist03.txt', 'testlist03.txt'], ['debug_trainlist.txt', 'testlist03.txt'], ['debug_trainlist_2.txt', 'testlist03.txt']]
#NUM_FRAMES = 64
PATH_TO_FRAMES_HMDB51 = '/home/users/datasets/hmdb51_opticalflow/'


def error(f):
        fil = open('extraction_fail.txt', 'a')
        fil.write(f+'\n')
        fil.close()

def resize_frames(frames, minsize=256):
        resized_frames = []
        frames = np.squeeze(frames)
        for frame in frames:
                #frame = cv2.resize(np.array(frame), (700, 600))
                if len(frame.shape)==3:
                        if frame.shape[1] >= frame.shape[0]:
                                if frame.shape[0] != minsize:
                                        scale = float(minsize) / float(frame.shape[0])
                                        frame = np.array(cv2.resize(frame, (int(frame.shape[1] * scale + 1), minsize))).astype(np.float32)
                        else:
                                if frame.shape[1] != minsize:       
                                        scale = float(minsize) / float(frame.shape[1])
                                        frame = np.array(cv2.resize(frame, (minsize, int(frame.shape[0] * scale + 1)))).astype(np.float32)
                resized_frames.append(frame)
        resized_frames = np.array(resized_frames)
        return resized_frames

def random_flip(frames):
        rand = randint(0,9)
        if rand < 5:
                #print 'flipping'
                flipped = []
                for frame in frames:
                        frame = cv2.flip(frame,1)
                        flipped.append(frame) 

                flipped = np.array(flipped)
                return flipped
        else:        
                return frames

def random_crop(frames, crop_size = 224):
        crop_x = randint(0, int(frames.shape[1] - crop_size))
        crop_y = randint(0, int(frames.shape[2] - crop_size))
        frames = frames[:, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        return frames

def center_crop(frames, crop_size = 224):
        crop_x = int((frames.shape[1] - crop_size) / 2)
        crop_y = int((frames.shape[2] - crop_size) / 2)
        frames = frames[:, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        return frames

def random_temporal_crop(filenames, min_video_length, extra_frames): 
        desired_frames = min_video_length + extra_frames
        if (len(filenames) < desired_frames): # in this case we must loop through the video to ensure there is a sufficient amount of frames. Extra frames gives a   margin for the temporal crop.
                for a in itertools.cycle(filenames):
                        filenames.append(a)
                        if (len(filenames)>=desired_frames):  # give the video extra frames for the random temporal sampling
                                break
        # apply the random temporal cropping
        starting_frame = randint(0,len(filenames) - min_video_length)
        filenames = filenames[starting_frame:starting_frame+min_video_length]
        return filenames

def center_temporal_crop(filenames, min_video_length): 
        desired_frames = min_video_length
        if (len(filenames) < desired_frames): # in this case we must loop through the video to ensure there is a sufficient amount of frames. 
                for a in itertools.cycle(filenames):
                        filenames.append(a)
                        if (len(filenames)>=desired_frames):  
                                break
        # apply the center temporal cropping

        starting_frame = int((len(filenames) - min_video_length) / 2) 
        filenames = filenames[starting_frame:starting_frame+min_video_length]

        return filenames


def read_rgb_images(files, is_training):
        images = []
        if not files:
                return 0
        for f in files:
                try:
                        current_image = cv2.imread(f)
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) # input uses RGB images
                        current_image = np.expand_dims(current_image,axis=0)
                        images.append(current_image)
                except:
                        error(f)
                        return 0
        images = np.vstack(images)
        images = np.expand_dims(images,axis=0)

        #perform augmentation here
        images = rescale(images)
        images = resize_frames(images)
        if is_training == True:
                images = random_crop(images, crop_size = 224)
                images = random_flip(images)
        else:
                images = center_crop(images, crop_size = 224)
        images = np.expand_dims(images,axis=0)
        return images

def read_flow_images(files_m, is_training):
        images = []
        for f in files_m:
                current_image = cv2.imread(f,1)
                current_image = np.expand_dims(current_image,axis=0)
                images.append(current_image)
        
        images = np.vstack(images)
        images = np.expand_dims(images,axis=0)
        #images = np.expand_dims(images,axis=-1)
        #perform augmentation here
        #print images[:,:,:,:,2]
        '''
        for i in range(3):
                for j in range(64):
                        cv2.imwrite('debug_images/flowdebug/'+str(i)+'_'+str(j)+'.jpg', images[0,j,:,:,i])
                        cv2.imwrite('debug_images/flowdebug/'+'full'+str(j)+'.jpg', images[0,j,:,:,:])

        raw_input('parou')
        '''
        images = rescale(images)
        images = resize_frames(images)
        if is_training == True:
                images = random_crop(images, crop_size=224)
                images = random_flip(images)
        else:
                images = center_crop(images, crop_size=224)
        images = np.expand_dims(images,axis=0)
        #images = np.swapaxes(images,)
        flow_x = np.expand_dims(images[:,:,:,:,2], axis=-1)
        flow_y = np.expand_dims(images[:,:,:,:,1], axis=-1)
        #return images[:,:,:,:,1:3]
        #print flow_x.shape, flow_y.shape
        return np.concatenate((flow_x,flow_y), axis=-1)

def read_flow_images_old(files_x, files_y, is_training):
        images_x = []
        images_y = []
        for f in files_x:
                current_image = cv2.imread(f,0)
                current_image = np.expand_dims(current_image,axis=0)
                images_x.append(current_image)
        for f in files_y:
                current_image = cv2.imread(f,0)
                current_image = np.expand_dims(current_image,axis=0)
                images_y.append(current_image)

        images_x = np.vstack(images_x)
        images_x = np.expand_dims(images_x,axis=0)
        images_x = np.expand_dims(images_x,axis=-1)

        images_y = np.vstack(images_y)
        images_y = np.expand_dims(images_y,axis=0)
        images_y = np.expand_dims(images_y,axis=-1)

        images = np.concatenate((images_x, images_y), axis=-1)

        #perform augmentation here
        images = rescale(images)
        images = resize_frames(images)
        if is_training == True:
                images = random_crop(images, crop_size=224)
                images = random_flip(images)
        else:
                images = center_Crop(images, crop_size=224)
        images = np.expand_dims(images,axis=0)
        #print images, images.shape
        return images




def save_batch(batch, labels, flow = False):
        save_dir = 'debug_images/'
        #print batch.shape
        for b, l in zip(batch, labels):
                #print b.shape
                try:
                        os.makedirs(save_dir+str(l))
                except:
                        pass
                for i,_b in enumerate(b):
                        if not flow:
                                _b = (_b + 1) * 128.
                                #print _b.shape
                                _b = _b[:,:,::-1]
                                cv2.imwrite(save_dir+str(l)+'/'+str(i)+'.jpg',_b)
                        else:
                                newdim = np.zeros((_b.shape[0], _b.shape[1], 1))
                                newdim[...] = 128
                                #print _b.shape
                                _b = (_b + 1) * 128.
                                _b = np.concatenate((newdim, _b), axis=-1)
                                _b = _b[:,:,::-1]
                                cv2.imwrite(save_dir+str(l)+'/'+str(i)+'.jpg',_b)

def rescale(frames):
        # nao sei o que eles querem dizer com truncate -20 20, tem que ver com o Wagner
        frames = frames / 128.0 - 1.0
        #flow_frame = -1 + 2.*(flow_frame - np.amin(flow_frame))/(np.amax(flow_frame) - np.amin(flow_frame))
        return frames
    
'''    
a=0
flow = True
for batches, labels in generate_train_batches(split_id = 3, flow = flow):
        print a, ': ', batches, batches.shape, labels, labels.shape
        save_batch(batches, labels, flow)
        #raw_input('jeje')
        a+=1
'''

def get_label(folder, class_dict):
        folder = folder.split('/')[0]
        return class_dict.get(folder)
        

class DataGenerator(keras.utils.Sequence):
    def _init_(self, list_IDs, class_dict, batch_size = 6,
                 n_classes = 101, flow = False, shuffle = True, is_training = True, num_frames = 64, hmdb_51 = False):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.flow = flow
        self.on_epoch_end()
        self.class_dict = class_dict
        self.is_training = is_training
        self.num_frames = num_frames
        self.hmdb = hmdb_51


    def _len_(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def _getitem_(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #print list_IDs_temp
        # Generate data

        #print 'fetching data...'
        #start = time.time()

        X,f,y = self.__data_generation(list_IDs_temp, self.flow)

        #finish = time.time()
        #print 'elapsed: ', finish - start
        return X, f, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames, flow = False):
        batch = []
        labels = []
        for video in filenames:
                folder = video.split('.')[0]
                l_ = get_label(folder, self.class_dict)
                if l_ is None:
                        l_ = -1
                labels.append(l_)
                   
                if self.flow == False:
                        files = sorted(glob(PATH_TO_FRAMES+folder+'/i_*'))
                        if self.hmdb:
                                files = sorted(glob(PATH_TO_FRAMES_HMDB51+folder+'/i_*'))  
                        if self.is_training == True:
                                files = random_temporal_crop(files, self.num_frames, 30)
                        else:
                                files = center_temporal_crop(files, self.num_frames)                                
                        images = read_rgb_images(files, self.is_training)
                        batch.append(images)
                else:
                        files = sorted(glob(PATH_TO_FRAMES+folder+'/m_*'))
                        if self.hmdb:
                                files = sorted(glob(PATH_TO_FRAMES_HMDB51+folder+'/m_*'))  
                        if self.is_training == True:
                                files = random_temporal_crop(files, self.num_frames, 30)
                        else:
                                files = center_temporal_crop(files, self.num_frames)                                
                        images = read_flow_images(files, self.is_training)
                        batch.append(images)

        labels = np.array(labels).astype(np.int)
        batch = np.vstack(batch)
        return batch,folder,labels

















'''


def read_batch(filenames, flow = False):
        batch = []
        labels = []
        for video in filenames:
                folder = video.split('.')[0]
                label = video.split(' ')[1]
                rgb_files = sorted(glob(PATH_TO_FRAMES+folder+'/i_*'))
                #if not rgb_files:
                #        print folder
                #        raw_input('vazio')
                flow_x_files = sorted(glob(PATH_TO_FRAMES+folder+'/x_*'))
                #flow_y_files = sorted(glob(PATH_TO_FRAMES+folder+'/y_*'))
                files = [rgb_files, flow_x_files, flow_y_files]
                files = random_temporal_crop(files, NUM_FRAMES, 30)
                if flow:
                        #images = read_flow_images(files[1], files[2])
                        images = read_flow_images(files[1])
                        batch.append(images)
                else:
                        images = read_rgb_images(files[0])
                        batch.append(images)
                labels.append(label)
        labels = np.array(labels).astype(np.int)
        batch = np.vstack(batch)
        return (batch,to_categorical(labels-1, num_classes=101))

def generate_train_batches(split_id = 0, batch_size = 5, shuffle = True, flow = False):
        current_idx = 0
        with open(PATH_TO_TRAIN_TEST_SPLITS+TRAIN_TEST_SPLITS_FILENAMES[split_id][0]) as f:
                lines = f.read().splitlines()
        if shuffle:
                from random import shuffle
                shuffle(lines)
        for i in range(len(lines)/batch_size):
                #print 'fetching data...'
                #start = time.time()
                yield read_batch(lines[current_idx:current_idx+batch_size], flow)
                #finish = time.time()
                #print 'elapsed: ', finish - start
                current_idx += batch_size
        #print 'remaining data'
        #yield read_batch(lines[len(lines) - len(lines)%batch_size:], flow)
