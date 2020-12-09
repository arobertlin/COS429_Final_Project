'''
This program demonstrates how to train and use a deep neural network with
2D Convolutional and recurrent layers, for activity recognition in videos.
I selected the KTH dataset since it is relatively small and has activities
that are easy to learn.

Chamin Morikawa
Last updated 2017-04-15
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
import seaborn as sns

# for file operations
import os
from PIL import Image

import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU # you can also try using GRU layers
from keras.optimizers import RMSprop, Adadelta, Adam, SGD # you can try all these optimizers
from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.layers.core import Activation, Dropout, Reshape, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from random import randint
import gc
import pandas as pd
import matplotlib.pyplot as plt
# natural sorting
import re

model_name = "3D_motion_cuboid_model"

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# use this if you want reproducibility
#np.random.seed(2016)

# we will be using tensorflow
# K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

# specifiy the path to your KTH data folder
trg_data_root = "KTH_Data_3D_motion_cuboid/"

# load training or validation data
# with 25 persons in the dataset, start_index and finish_index has to be in the range [1..25]
def load_data_for_persons(root_folder, start_index, finish_index, frames_per_clip):
    # these strings are needed for creating subfolder names
    class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"] # 6 labels
    frame_path = "/frames2/"
    frame_set_prefix = "person" # 2 digit person ID [01..25] follows
    rec_prefix = "d" # seq ID [1..4] follows
    rec_count = 4
    seg_prefix = "seg" # seq ID [1..4] follows
    seg_count = 4

    data_array = []
#     data_array_np = np.zeros((frames_per_clip, rec_count*seg_count*25*len(class_labels), 120, 160))
    data_array_np = np.zeros((rec_count*seg_count*(finish_index-start_index+1)*len(class_labels), frames_per_clip, 120, 160, 1))
    
    classes_array = []
    z = 0

    # let's make a couple of loops to generate all of them
    for i in range(0, len(class_labels)):
        # class
        class_folder = trg_data_root + class_labels[i] + frame_path

        for j in range(start_index, finish_index+1):
            # person
            if j<10:
                person_folder = class_folder + frame_set_prefix + "0" + str(j) + "_" + class_labels[i] + "_"
            else:
                person_folder = class_folder + frame_set_prefix + str(j) + "_" + class_labels[i] + "_"

            for k in range(1,rec_count+1):
                # recording
                rec_folder = person_folder + rec_prefix + str(k) + "/"
                for m in range(1,seg_count+1):
                    # segment
                    seg_folder = rec_folder + seg_prefix + str(m) + "/"

                    # get the list of files
                    file_list = [f for f in os.listdir(seg_folder)]
                    example_size = len(file_list)

                    # for larger segments, we can change the starting point to augment the data
                    clip_start_index = 0
                    if example_size > frames_per_clip:
                        # set a random starting point but fix length - augments data, but slows training
                        #clip_start_index = randint(0, (example_size - frames_per_clip))
                        # sample the frames from the center
                        clip_start_index = int(example_size/2 - frames_per_clip/2)
                        example_size = frames_per_clip

                    # need natural sort before loading data
                    file_list.sort(key=natural_sort_key)

                    #create a list for each segment
                    current_seg_temp = []
                    for n in range(clip_start_index,example_size+clip_start_index):
                        file_path = seg_folder + file_list[n]
#                         print(file_path)
                        data = np.asarray( Image.open(file_path), dtype='uint8' )
                        # remove unnecessary channels
#                         data_gray = np.delete(data,[1,2],2)
#                         data_gray = data_gray.astype('float32')/255.0
#                         current_seg_temp.append(data_gray)
                        data_array_np[z,n-clip_start_index,:,:,:] = data.reshape(120,160,1)

                    # preprocessing
                    current_seg = np.asarray(current_seg_temp)
                    current_seg = current_seg.astype('float32')

                    data_array.append(current_seg)
                    classes_array.append(i)
                    z = z+1

    # # create one-hot vectors from output values
    classes_one_hot = np.zeros((len(classes_array), len(class_labels)))
    classes_one_hot[np.arange(len(classes_array)), classes_array] = 1

    # done
    print(data_array_np.shape)
    return (data_array_np, classes_one_hot)

# what you need to know about data, to build the model
img_rows = 120
img_cols = 160
maxToAdd = 25 # use 25 consecutive frames from each video segment, as a training sample
nb_classes = 6
class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# build network model
print("Building model")

#define our time-distributed setup
model = Sequential()

# note: the architecture below is just for demonstration.
# for higher accuracy, you will have to change the number and dimensions of layers.

# three convolutional layers
# model.add(TimeDistributed(Conv2D(4, 5, strides=(2, 2), padding='valid'), input_shape=(maxToAdd,img_rows,img_cols,1)))
# model.add(Activation('relu'))

# model.add(TimeDistributed(MaxPooling2D(strides=(2, 2)),input_shape=(maxToAdd,img_rows,img_cols,1)))

model.add(TimeDistributed(Conv2D(16, (7,9), strides=(1,1), padding='valid'), input_shape=(maxToAdd,img_rows,img_cols,1)))
model.add(Activation('relu'))

model.add(TimeDistributed(MaxPooling2D(strides=(3, 3))))

model.add(TimeDistributed(Conv2D(64, 7, strides=(1, 1), padding='valid')))
model.add(Activation('relu'))

model.add(TimeDistributed(MaxPooling2D(strides=(3, 3))))

# model.add(TimeDistributed(Conv2D(64, (4,6), strides=(1, 1), padding='valid')))
# model.add(Activation('relu'))

# flatten and prepare to go for recurrent learning
model.add(TimeDistributed(Flatten()))

# a single dense layer
model.add(TimeDistributed(Dense(80)))
model.add(BatchNormalization()) # required for ensuring that the network learns
model.add(Activation('relu'))

# GRU layers
#model.add(GRU(output_dim=100,return_sequences=True))
#model.add(GRU(output_dim=50,return_sequences=False))

# the LSTM layer performed better than GRU layers
model.add(LSTM(80, activation = 'tanh'))

# let's try some dropping out here
model.add(Dropout(.5))

# fully connected layers to finish off
model.add(Dense(80, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

print(model.summary())

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) #adam is faster, but you can use the others too.

#training parameters
batch_size = 16 # increase if your system can cope with more data
nb_epochs = 5 # I once achieved 77.5% accuracy with 100 epochs. Feel free to change

print ("Loading data")
# load training data
X_train, y_train = load_data_for_persons(trg_data_root, 1, 15, maxToAdd)

X_val, y_val = load_data_for_persons(trg_data_root, 16, 18, maxToAdd)
val_data = (X_val, y_val)
# NOTE: if you can't fit all data in memory, load a few users at a time and
# use multiple epochs. I don't recommend using one user at a time, since
# it prevents good shuffling.

# perform training
print("Training")
# X_train = tf.convert_to_tensor(X_train)
# X_train = np.asarray(X_train).astype(np.float32)
history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=val_data, validation_freq=1, epochs=nb_epochs, shuffle=False, verbose=1)

# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_name + "_acc")
plt.show()


# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(model_name + "_loss")
plt.show()

# clean up the memory
X_train       = None
y_train       = None
X_val = None
y_val = None
gc.collect()

print("Testing")

# load test data: in this case, person 9
X_test, y_test = load_data_for_persons(trg_data_root, 19, 25, maxToAdd)
print('Total no. of testing samples used:', y_test.shape[0])

preds = model.predict(tf.convert_to_tensor(X_test))
print(len(preds))

# confusion_matrix = np.zeros(shape=(y_test.shape[1],y_test.shape[1]))

accurate_count = 0.0
for i in range(len(preds)):
    # if you are not sure of the axes of the confusion matrix, try the following line
#     print ('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_val_one_hot[i])))

    # calculating overall accuracy
    if np.argmax(preds[i])==np.argmax(np.array(y_test[i])):
        accurate_count += 1
        
        
print('Test accuracy: ', 100*accurate_count/len(preds)),' %'

# Make confusion matrix
classes = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
confusion_matrix = np.zeros(shape=(y_test.shape[1],y_test.shape[1]))
accurate_count = 0.0
for i in range(0,len(preds)):
    # updating confusion matrix
    confusion_matrix[np.argmax(preds[i])][np.argmax(np.array(y_test[i]))] += 1

    # if you are not sure of the axes of the confusion matrix, try the following line
    #print ('Predicted: ', np.argmax(preds[i]), ', actual: ', np.argmax(np.array(y_val_one_hot[i])))

    # calculating overall accuracy
    if np.argmax(preds[i])==np.argmax(np.array(y_test[i])):
        accurate_count += 1

print('Confusion matrix:')
print(class_labels)
print(confusion_matrix)
# print(y_test.shape)
# print(preds.shape)
# con_mat = tf.math.confusion_matrix(labels = y_test, predictions = preds).numpy()

con_mat_norm = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
print(con_mat_norm.shape)

con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)

figure = plt.figure(figsize = (7, 7))
sns.heatmap(con_mat_df, annot=True, cmap = plt.cm.Blues)
plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(model_name + "_confusion")



# print('Confusion matrix:')
# print(class_labels)
# print(confusion_matrix)

#save the model
model.save(model_name)

# done.
