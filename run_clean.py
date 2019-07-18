import numpy as np
import tensorflow as tf
from tensorflow import keras

from main import *
from helper import *

''' Script to train recon on undistorted (clean) images with different weightings on loss functions '''

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
    #dft_mat = fft2(I)
    dft_real = np.real(dft_mat)
    dft_imag = np.imag(dft_mat)
    Phi_input = np.concatenate((dft_real, dft_imag), axis=1)
    #Phi_input = np.random.normal(size=(N,2*N))
        
	# Classifier
	#x_train, y_train, x_test, y_test = load_data()
	#mnist_class = fit_classifier(x_train,y_train,x_test,y_test,batch_size=64, epochs=12, trainable=False)
	#mnist_class.save('mnist_classifier2.h5')
    #(sess, Prediction,Pre_symetric, classified, clean_images, Phi_input, X_input) = train(name, Phi_input, loss_alpha = 1,loss_beta = 1,loss_gamma = 0, Training_inputs=None, classifier_trainable=False)
    

    # Not distorted
    loss_alpha = 0
    loss_beta = 1
    loss_gamma = 10
    name = 'run'+'_alpha_%d_beta_%d_gamma_%d' % (loss_alpha, loss_beta, loss_gamma)

    #train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=None, classifier_trainable=False)
    l = [0, 1, 10]
    for loss_alpha in l:
        for loss_beta in l:
            for loss_gamma in l:
                name = 'run'+'_alpha_%d_beta_%d_gamma_%d' % (loss_alpha, loss_beta, loss_gamma)
                if loss_alpha != 0 or loss_beta != 0 or loss_gamma != 0:
                    train(name, Phi_input, loss_alpha = loss_alpha, loss_beta = loss_beta, loss_gamma = loss_gamma, Training_inputs=None, classifier_trainable=False)

if __name__ == '__main__':
    main()
