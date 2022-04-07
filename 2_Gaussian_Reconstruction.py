from scipy import ndimage
from scipy.interpolate import griddata
from skimage.measure import regionprops
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math




#------NORMALISE 256 MASK AVERAGED------#
def Normalisation(image):
    #Normalise an image using openCV
    downscallingFactor = 4
    sigmaGauss = 20

    lowsat = 0.3 #%
    uppersat = 99 #%

    if(sigmaGauss/downscallingFactor > 2):
        kernalDiamCalculated = round((sigmaGauss/downscallingFactor * 8 + 1))
        if(kernalDiamCalculated % 2 == 0):
            kernalDiamCalculated += 1
    else:
        kernalDiamCalculated = 2 * math.ceil(2 * sigmaGauss/downscallingFactor) + 1

    imageNormOutput = cv2.GaussianBlur(image, (kernalDiamCalculated,kernalDiamCalculated), sigmaGauss/downscallingFactor)
    
    #lowerBoundaryInput = np.percentile(imageNorm, lowsat)
    #lowerBoundaryOutput = 0
    #upperBoundayInput = np.percentile(imageNorm, uppersat)
    #upperBoundaryOutput = 255
    #imageNormOut = (imageNorm - lowerBoundaryInput) * ((upperBoundaryOutput - lowerBoundaryOutput) / (upperBoundayInput - lowerBoundaryInput)) + lowerBoundaryOutput

    imageNormOutput *= 2

    return imageNormOutput




#------INTERPOLATE 256 MASK AVERAGED------#
def Interpolation(image):
    #Use interpolation to smooth image
    shrinkFactorContrastMask = 0.9
    lowerSet = 0.3
    upperSat = 99.7

    rows = image.shape[0]
    cols = image.shape[1]

    binaryCores = image > 0.0
    labels, nlabels = ndimage.label(binaryCores)


    props = regionprops(labels, intensity_image = image)
    
    xCoords = []
    yCoords = []
    #coreSizeVector = []
    values = []
    for region in props:
        centre_coords = np.around(region.centroid,0)
        xCoords.append(round(centre_coords[1],0))
        yCoords.append(round(centre_coords[0],0))

        values.append(region.mean_intensity)

    x = np.linspace(1,256,256)
    y = np.linspace(1,256,256)

    Xgrid, Ygrid = np.meshgrid(x,y)

    coords = np.vstack((xCoords, yCoords)).T

    delaunTri = Delaunay(coords)
    #plt.triplot(coords[:,0], coords[:,1], delaunTri.simplices)
    #plt.scatter(xCoords, yCoords, marker = "+")
    #plt.imshow(image)
    #plt.show()


    #gridzNear = griddata(coords, values, (Xgrid, Ygrid), method='nearest')
    gridzLin = griddata(coords, values, (Xgrid, Ygrid), method='linear')
    #gridzCube = griddata(coords, values, (Xgrid, Ygrid), method='cubic')

    #fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    #ax[0,0].set_title("Binary core image")
    #ax[0,0].imshow(image)

    #ax[0,1].set_title("Nearest")
    #ax[0,1].imshow(gridzNear, cmap = 'gray')

    #ax[1,0].set_title("Linear")
    #ax[1,0].imshow(gridzLin, cmap = 'gray')

    #ax[1,1].set_title("Cubic")
    #ax[1,1].imshow(gridzCube, cmap = 'gray')
    #plt.show()

    return gridzLin


#imagePath = "./imageComparison/"

#image = cv2.imread(imagePath + "LR0.png", 0)

#normImage = Normalisation(image)
#interpImage = Interpolation(image)

#cv2.imwrite(imagePath + "normalisation.png", normImage)
#cv2.imwrite(imagePath + "interpolation.png", interpImage)






testing_datasets = sorted(glob(dataset_dir +"/imagedata_testing*"))

print("Creating testing slices from test originals")
for image_path in tqdm(testing_datasets):
#Load in original images
#image_path = Test_images[0]
    dataset_image = cv2.imread(image_path)

    dataset_count = utils.div_original(dataset_dir, dataset_count, dataset_image, train_set = False)



cv2.waitKey(0)