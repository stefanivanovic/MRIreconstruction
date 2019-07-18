#MRIdist2.py
from tensorflow import keras
import numpy as np
import math
import copy
from scipy.ndimage.filters import uniform_filter

#NOTE: Editable Paremeters will be marked "Changable"

def fft2c(x):
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))

def makeRadialTrajectory(shape1, shape2):
    y1 = np.tile(np.arange(shape2[0]), shape2[1])
    x1 = np.repeat(np.arange(shape2[1]), shape2[0])
    valueScaling = ((2 / shape2[1]) * (x1+1))
    rSpace = (shape1[0] - 1) / (2 * shape2[0])
    thetaSpace = (2 * math.pi) / shape2[1]
    rDists = rSpace * (x1+1)
    thetaDists = thetaSpace * y1
    center = [(shape1[0])/2 , (shape1[1])/2] #The center is technicaly (shape1[0]-1)/2, but we remove the -1 since "int" rounds down.
    x2 = (rDists * np.cos(thetaDists)) + center[0]
    y2 = (rDists * np.sin(thetaDists)) + center[1]
    x2 = x2.astype(int)
    y2 = y2.astype(int)
    thoughtK = np.transpose([y2, x2])
    return (thoughtK, thoughtK, valueScaling)

def makeCartesianTrajectory(shape1, shape2, randomLine, ShiftError):
    x1 = (np.arange(shape2[1]) * (shape1[1] / shape2[1])).astype(int)
    valueScaling = np.ones(shape2[0] * shape2[1])
    x2 = np.tile(x1, shape2[0])
    if randomLine:
        y1 = np.random.choice(shape1[0], shape2[0], replace=False)
    else:
        y1 = (np.arange(shape2[0]) * (shape1[0] / shape2[0])).astype(int)

    thoughtK = np.transpose([np.repeat(y1, shape2[1]), x2])

    if ShiftError:
        ErrorFactor = 0.05 #Changable
        ErrorAmount = shape1[0] * ErrorFactor
        realY = (y1 + (np.random.normal(size=shape2[0]) * ErrorAmount)).astype(int)
        realK = np.transpose([np.repeat(realY, shape2[1]), x2])
        return (thoughtK, realK, valueScaling)
    else:
        return (thoughtK, thoughtK, valueScaling)

def getTrajectory(shape1, shape2, radial, randomLine, ShiftError):
    if radial:
        (thoughtK, realK, valueScaling) = makeRadialTrajectory(shape1, shape2)
    else:
        (thoughtK, realK, valueScaling) = makeCartesianTrajectory(shape1, shape2, randomLine, ShiftError)
    valueScaling = valueScaling * ((shape1[0] * shape1[1]) / (shape2[0] * shape2[1]))
    return (thoughtK, realK, valueScaling)

def coordinatesToImage(coordinates, shape1):
    [y1, x1] = coordinates.T
    keys = (y1 * shape1[0]) + x1
    image_1 = np.zeros(shape1[0] * shape1[1])
    image_1[keys] = 1
    image_1 = np.split(image_1, shape1[0])
    return image_1

def makeDistImage(realK, thoughtK, image1, shape2, valueScaling):
    shape1 = image1.shape
    [y1, x1] = realK.T
    keys1 = (y1 * shape1[0]) + x1
    [y2, x2] = thoughtK.T
    keys2 = (y2 * shape1[0]) + x2
    keys3, indexs = np.unique(keys2, return_index=True)
    image1Array = np.concatenate(image1)
    values = image1Array[keys1] * valueScaling
    if len(keys3) < len(keys2):
        values = [np.sum(values[np.argwhere(keys2==i)]) for i in keys3]
        keys2 = keys3
    newImage = np.zeros(shape1[0] * shape1[1]).astype(complex)
    newImage[keys2] = values
    newImage = np.array(np.split(newImage, shape1[0]))
    return newImage

def implementRealOnFullSpace(image1, realK, shape1):
    [y1, x1] = realK.T
    keys1 = (y1 * shape1[0]) + x1
    values = np.concatenate(image1)
    newImage = np.zeros(shape1[0] * shape1[1]).astype(complex)
    newImage[keys1] = values
    newImage = np.array(np.split(newImage, shape1[0]))
    return newImage

def inhomoFT(img, realK, shape1, shape2, MagSus, FieldCon):
    def getNaiveGradients(pos, shape2, timeDelays):
        pos = np.insert(pos, 0, 0)
        lineFactor = np.tile(np.insert(np.ones(shape2[1]-1),0, 0), shape2[0])
        phaseChange = pos[1:] - (pos[:-1] * lineFactor)
        grads = phaseChange / timeDelays
        return grads

    B0 = 1 #TODO: Make Realistic
    img1 = np.concatenate(img)
    if MagSus:
        magSupt = makeMagSupt(img)
        ms = np.concatenate(magSupt)
        img1 = img1 * ms
    else:
        ms = np.ones(len(img1))
    x1 = np.tile(np.arange(shape1[0]) / (shape1[0]), shape1[1]) - .5
    z1 = np.repeat(np.arange(shape1[1]) / (shape1[1]), shape1[0]) - .5
    [zPos, xPos] = realK.T
    zPos = (zPos/shape1[0]) - .5
    xPos = (xPos/shape1[0]) - .5
    timeDelays = np.ones(shape2[0] * shape2[1]) * 0.0002 / 0.06
    gamma = 7622.593285 #Hz/T
    timeDelays = timeDelays*gamma
    xGrad = getNaiveGradients(xPos, shape2, timeDelays)
    zGrad = getNaiveGradients(zPos, shape2, timeDelays)
    f1 = ((zGrad**2) / (8 * B0)) * timeDelays
    f2 = ((xGrad**2) / (2 * B0)) * timeDelays
    f3 = ((xGrad * zGrad) / (2 * B0)) * timeDelays
    f4 = zGrad * timeDelays
    f5 = xGrad * timeDelays
    Mat4 = np.outer(f4, z1*ms) #Regular Field
    Mat5 = np.outer(f5, x1*ms) #Regular Field
    PhaseShiftMatrix = Mat4 + Mat5
    if FieldCon:
        Mat1 = np.outer(f1, (x1**2)*ms) #Non-Linearity
        Mat2 = np.outer(f2, (z1**2)*ms) #Non-Linearity
        Mat3 = np.outer(f3, (x1*z1)*ms) #Non-Linearity
        PhaseShiftMatrix = PhaseShiftMatrix + Mat1 + Mat2 + Mat3
    #'''
    if MagSus:
        B0Field = np.ones(len(xGrad)) * B0 * timeDelays
        Mat6 = np.outer(B0Field, np.ones(len(ms))-ms)#B0
        PhaseShiftMatrix = PhaseShiftMatrix + Mat6
    #'''
    #PhaseShiftMatrix = np.outer(zGrad, z1) + np.outer(xGrad, x1)
    #PhaseShiftMatrix = np.array([np.arange(6),np.arange(6)]).T
    PhaseShiftSplit = np.split(PhaseShiftMatrix, shape2[0])
    PhaseMatrixSplit = np.cumsum(PhaseShiftSplit, axis=1) #partial addition not full addition should be used.
    PhaseMatrix = np.concatenate(PhaseMatrixSplit, axis=0) * shape1[0]
    #PhaseMatrix2 = np.outer(zPos, z1) + np.outer(xPos, x1)
    Fourier = np.exp(-2*np.pi*1j*PhaseMatrix)
    #print (Fourier[1])
    imgFourier = np.matmul(Fourier, img1) / ((shape1[0]*shape1[1])**0.5)
    imgFourier1 = np.array(np.split(imgFourier, shape2[0]))
    return imgFourier1

def makeMagSupt(image):
    image4 = copy.copy(image).astype(float)
    image4[image4 > 0.1] = 1.0
    image4[image4 != 1] = 0.0
    size1 = 2 #Changable
    image5 = uniform_filter(image4, size=size1)#[::size1,::size1]
    blurr = 1.0
    image4 = (((1-blurr) * image4) + (blurr * image5)).astype(float)
    return image4

def makeDistorter(shape1, shape2, DistortionChoices):
    DistortionChoices = (radial, randomLine, ShiftError, MagSus, FieldCon)


# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# img = train_images[1]
# shape1 = img.shape
# yUnderSample = 2 #Changable
# xUnderSample = 2 #Changable
# shape2 = (int(shape1[0]/yUnderSample), int(shape1[1]/xUnderSample))
# radial = False #Changable
# randomLine = False #Changable
# ShiftError = False #Changable
# MagSus = False #Changable
# FieldCon = False #Changable
# DistortionChoices = (radial, randomLine, ShiftError, MagSus, FieldCon)
# (thoughtK, realK, valueScaling) = getTrajectory(shape1, shape2, radial, randomLine, ShiftError)
# if MagSus or FieldCon:
#     image1 = inhomoFT(img, realK, shape1, shape2, MagSus, FieldCon)
#     image1 = implementRealOnFullSpace(image1, realK, shape1)
# else:
#     image1 = fft2c(img)

# #image1 is on grid but realy in k space.
# image2 = makeDistImage(realK, thoughtK, image1, shape2, valueScaling)
# img2 = ifft2c(image2)
# image3 = coordinatesToImage(thoughtK, shape1)

# plotImages([img, img2, image3])
