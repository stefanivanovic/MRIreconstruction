import keras
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
#import sigpy as sp
from scipy.linalg import dft
from MRIdist3 import makeCartesianTrajectory, inhomoFT, fft2c, ifft2c

num_classes=10
img_rows, img_cols = 28, 28

# print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    input_shape = (1,img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols,1)

print(input_shape)

num_classes = 10
n_output = 784
batch_size = 16
PhaseNumber = 4 #Number of "Layers" in istanet
#nrtrain = 10000 #Training samples in an epoch
nrtrain = 40000 #Training samples in an epoch
learning_rate = 0.001
#EpochNum = 2
EpochNum = 15
TRAIN = 1 #Train or test boolean variable
log_every_k_batches = 40 #How many batches should tensorboard logging take

def load_data():
    # input image dimensions

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def distort_data(inp):
    imgs_con = np.zeros(inp.shape, dtype="complex")
    imgs_sus = imgs_con 
    #ksps_con = np.zeros((inp.shape[0],784*2,1), dtype="complex")
    ksps_con = np.zeros((inp.shape[0],784*2,1))
    ksps_sus = ksps_con

    for i in range(inp.shape[0]):
        img = inp[i,:,:]
        shape1 = img.shape
        shape2 = shape1 # Fully sampled
        # Cartesian
       
        randomLine = False
        ShiftError = False

        (thoughtK, realK, valueScaling) = makeCartesianTrajectory(shape1, shape2, randomLine, ShiftError)
        ksp_con = inhomoFT(img, realK, shape1, shape2, MagSus=False, FieldCon=True) # kspace
        ksp_sus = inhomoFT(img, realK, shape1, shape2, MagSus=True, FieldCon=False) # kspace

        ksp_con = np.reshape(ksp_con,(784,1))
        ksp_sus = np.reshape(ksp_sus,(784,1))

        ksp_con_real = np.real(ksp_con)
        ksp_con_imag = np.imag(ksp_con)
        ksp_con = np.concatenate((ksp_con_real, ksp_con_imag), axis=0)
        #print(ksp_con.shape)

        ksp_sus_real = np.real(ksp_sus)
        ksp_sus_imag = np.imag(ksp_sus)
        ksp_sus = np.concatenate((ksp_sus_real, ksp_sus_imag), axis=0)

        ksps_con[i,:,:] = ksp_con
        ksps_sus[i,:,:] = ksp_sus

        #imgs_con[i,:,:] = np.abs(fft2c(ksp_con))
        #imgs_sus[i,:,:] = np.abs(fft2c(ksp_sus))
    

    #ksp_con_real = np.real(ksp_con)
    #ksp_con_imag = np.imag(ksp_con)
    #ksp_con = np.concatenate((ksp_con_real, ksp_con_imag), axis=1)
    #return(imgs_con, imgs_sus)
    #ksp_sus_real = np.real(ksp_sus)
    #ksp_sus_imag = np.imag(ksp_sus)
    #ksp_sus = np.concatenate((ksp_sus_real, ksp_sus_imag), axis=1)
    return(ksps_con, ksps_sus)


def fit_classifier(x_train,y_train,x_test,y_test, batch_size=64, epochs=12, trainable=False):
    model = keras.models.Sequential()
    # print(input_shape)
    # model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
    #           activation='relu',
    #           input_shape=(1,28,28)))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.trainable = trainable
    return model

def load_model(trainable = False):
    model = keras.models.load_model('mnist_classifier.h5')
    model.trainable = trainable
    return model
