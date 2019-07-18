from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def fft2c(x):
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))

def sim_dist(fov, ksp):
	# Px is pixelwise shift
	# Within plane shift --> phase error
	Gz = 4e-4 # T/cm
	B0 = 5e-3 # T

	shift = Gz/(2*B0)
	z = np.linspace(-1*fov, fov, len(ksp))
	ksp_shift = np.exp(2*np.pi*1j*shift*z**2)*ksp
	img_shift = fft2c(ksp_shift)

	# plt.figure()
	# plt.imshow(np.abs(img_shift),"gray")
	return ksp_shift