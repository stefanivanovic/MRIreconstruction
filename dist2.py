#MRIdist2.py
import numpy as np
import math
import copy

#NOTE: Editable Paremeters will be marked "Changable"

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
    x2 = np.repeat(x1, shape2[0])
    if randomLine:
        y1 = np.random.choice(shape1[0], shape2[0], replace=False)
    else:
        y1 = (np.arange(shape2[0]) * (shape1[0] / shape2[0])).astype(int)

    thoughtK = np.transpose([np.tile(y1, shape2[1]), x2])

    if ShiftError:
        ErrorFactor = 0.05 #Changable
        ErrorAmount = shape1[0] * ErrorFactor
        realY = (y1 + (np.random.normal(size=shape2[0]) * ErrorAmount)).astype(int)
        realK = np.transpose([np.tile(realY, shape2[1]), x2])
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

def makeDistImage(realK, thoughtK, image1, valueScaling):
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
    newImage = np.zeros(shape1[0] * shape1[1])
    newImage[keys2] = values
    newImage = np.split(newImage, shape1[0])
    return newImage