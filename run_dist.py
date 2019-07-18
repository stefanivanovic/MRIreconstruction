import numpy as np
import tensorflow as tf
from tensorflow import keras

from main import *
from helper import *

mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def fft2(x):
    ''' Centered 2D fft '''
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2(x):
    ''' Centered 2D inverse fft '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))

def main():
    name = 'run'
    model_dir = name+'HALF_Phase_%d_ratio_0_ISTA_Net_plus_Model' % (PhaseNumber)

    # Generate DFT matrix
    N = 784
    I = np.eye(N)
    mat = []
    for i in I:
        mat.append(fft2(i.reshape((28,28))).flatten())
    dft_mat = np.array(mat)
    print(dft_mat.shape)
    dft_real = np.real(dft_mat)
    dft_imag = np.imag(dft_mat)
    Phi_input = np.concatenate((dft_real, dft_imag), axis=1)
    
    #(ksp_con, ksp_sus) = distort_data(x_train[0:2,:,:])

    l = [0,1,10]

    # Clean images
    #clean_images = clean_images[1,:]

    #train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=None, classifier_trainable=False)
    l = [0, 1, 10]
    for loss_alpha in l:
        for loss_beta in l:
            for loss_gamma in l:
                name = 'run'+'_alpha_%d_beta_%d_gamma_%d' % (loss_alpha, loss_beta, loss_gamma)
                if loss_alpha != 0 or loss_beta != 0 or loss_gamma != 0:
                    train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=None, classifier_trainable=False)
   
    # Generate distorted images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (ksp_con, ksp_sus) = distort_data(x_train[0:nrtrain,:,:])
    print("Ksp con shape = {}".format(ksp_con.shape))
    print("Ksp sus shape = {}".format(ksp_sus.shape))
    
    # Distorted, concomitant
    for loss_alpha in l:
        for loss_beta in l:
            for loss_gamma in l:
                name = 'condist'+'_alpha_%d_beta_%d_gamma_%d' % (loss_alpha, loss_beta, loss_gamma)
                #(sess, Prediction,Pre_symetric, classified, clean_images, Phi_input, X_input) = train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=ksp_con, classifier_trainable=False)
                train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=ksp_con, classifier_trainable=False)
    # Distorted, susceptibility
    for loss_alpha in l:
        for loss_beta in l:
            for loss_gamma in l:
                name = 'susdist'+'_alpha_%d_beta_%d_gamma_%d' % (loss_alpha, loss_beta, loss_gamma)
                train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=ksp_sus, classifier_trainable=False)
if __name__ == '__main__':
    main()
