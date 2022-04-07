"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

#import cv2
from cv2 import imread, imwrite, imshow, circle, resize
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import shift, rotate, label
from skimage.transform import rescale
from skimage.feature import masked_register_translation
from scipy.interpolate import griddata
from skimage.measure import regionprops
from math import floor
from numpy.core.function_base import linspace


def load_original(image_path):
    """
    Function to read in original clipped images to allow for dividing into 256x256 size samples

    Parameters
    ----------
    base_path:
        path to image dataset folder
    dataset_number:
        name of original image 
    """
    image = imread(image_path)
    return image

def resize_image(image, x_size, y_size):
    resized_image = resize(image, (x_size, y_size), interpolation = cv2.INTER_AREA)


    return resized_image


def div_original(base_path, dataset_count, original_image, train_set):

    width = original_image.shape[1]
    height = original_image.shape[0]

    width_slices = round(width / 256)-1
    height_slices = round(height / 256)-1
    amount_slices = width_slices * height_slices

    slices = np.empty((256,256,amount_slices),dtype="uint16")

    count = 0
    for x in range(width_slices):
        for y in range(height_slices):
            xStart = x * 256
            xStop = xStart + 256
            yStart = y * 256
            yStop = yStart + 256

            slice = original_image[yStart:yStop,xStart:xStop]
            dataset_count = save_slice(base_path, dataset_count, slice, train_set)
            count += 1

    return dataset_count


def save_slice(base_path, dataset_count, image_slice, train_set):
    if(train_set):
        save_path = base_path + "/imagedata_training" + str(dataset_count) 
    else:
        save_path = base_path + "/imagedata_testing" + str(dataset_count) 


    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    imwrite(save_path + "/HR.png", image_slice)
    dataset_count+=1

    return dataset_count


def AverageCores(img):
    # use a boolean condition to find where pixel values are > 0
    blobs = img > 0.0
    # label connected regions that satisfy this condition
    labels, nlabels = label(blobs)
    props = regionprops(labels, intensity_image = img)
    means = []
    for region in props:
        means.append(region.mean_intensity)

    x = img.shape[0]
    y = img.shape[1]
    image = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            if(img[i,j] > 0):
                image[i,j] = means[labels[i,j]-1]
    
    return image


def create_circular_mask():
    circle_mask = np.zeros((256,256,3))
    radi = floor(256/2)
    circle(circle_mask, center=(radi,radi), radius=radi, color=(1,1,1), thickness= -1)
    return circle_mask

def create_circular_GT(ground, circle_mask, save_path): #ground_path = "HR.png"
    circle_ground = ground * circle_mask
    imwrite(save_path, circle_ground)

def create_circular_bundle(fibre_bundle, circle_mask, save_path): # fibre_mask = "thresh256_x1.tif"
    circle_bundle = fibre_bundle * circle_mask
    imwrite(save_path, circle_bundle)

def rotate_fibre(circle_ground_path, circle_bundle, rot_left, rot_right, rot_number, save_path): #circle_bundle_path = "./circle_bundle.png"
    circle_ground = imread(circle_ground_path)/256

    for rot in linspace(rot_left, rot_right, rot_number):
        bundle_rot = circle_bundle.rotate(rot)

        #Overlay bundle
        #bundle_image_u8 = (circle_ground * bundle_rot)/256
        bundle_image = (circle_ground * bundle_rot)

        #bundle_ave_u8 = AverageCores(bundle_image_u8)
        bundle_ave = np.zeros((bundle_image.shape[1],bundle_image.shape[0],bundle_image.shape[2]))

        for c in range(bundle_image.shape[2]):
            bundle_ave[:,:,c] = AverageCores(bundle_image[:,:,c])


        #imwrite(save_path + f'/LR{rot}.png', bundle_ave_u8)
        imwrite(save_path + f'/LR{rot}.png', bundle_ave)


#def overlay_mask(base_path, dataset_folderpath, img_slice, mask_name): #dataset_dir, folder, img_slice, mask_name, magnification
#    mask = imread(base_path + "/" + "masks" + "/" + mask_name + ".tif",0)

#    mask90 = ndimage.rotate(mask, 90)
#    mask180 = ndimage.rotate(mask, 180)
#    mask270 = ndimage.rotate(mask, 270)

#    img_slice = img_slice / 255

#    masked_slice = img_slice * mask

#    averaged_masked_slice = AverageCores(masked_slice)
#    imwrite(dataset_folderpath + "/LR0.png", averaged_masked_slice)


#    masked_slice90 = img_slice * mask90
#    averaged_masked_slice90 = AverageCores(masked_slice90)
#    imwrite(dataset_folderpath + "/LR90.png", averaged_masked_slice90)

#    masked_slice180 = img_slice * mask180
#    averaged_masked_slice180 = AverageCores(masked_slice180)
#    imwrite(dataset_folderpath + "/LR180.png", averaged_masked_slice180)

#    masked_slice270 = img_slice * mask270
#    averaged_masked_slice270 = AverageCores(masked_slice270)
#    imwrite(dataset_folderpath + "/LR270.png", averaged_masked_slice270)


def dataset_stack_4D(dataset, segments):

    X = np.empty((len(dataset),256,256,segments))
    Y = np.empty((len(dataset),256,256,segments))

    for image_number,folder_path in tqdm(enumerate(dataset)):

        LRs = sorted(glob(folder_path+"/LR*.png"))
        HR = sorted(glob(folder_path+"/HR.png"))

        L = len(LRs)

        LR_stack = np.empty((256,256,L*3))
        HR_stack = np.empty((256,256,L*3))

   
        for i,img in enumerate(LRs):
            LR_stack[:,:,i:i+3] = imread(img)/256 #256,256,Segments
            HR_stack[:,:,i:i+3] = imread(HR[0])/256 #256,256,Segments

        X[image_number,:,:,:] = LR_stack
        Y[image_number,:,:,:] = HR_stack

    return X, Y



def dataset_stack_4D_colour(dataset, segments):

    X = np.empty((len(dataset),256,256,segments * 3))
    Y = np.empty((len(dataset),256,256,3))

    for image_number,folder_path in tqdm(enumerate(dataset)):

        LRs = sorted(glob(folder_path+"/LR*.png"))
        HR = sorted(glob(folder_path+"/HR.png"))

        L = len(LRs)

        LR_stack = np.empty((256,256,L*3))
        HR_stack = np.empty((256,256,L*3))

   
        for i,img in enumerate(LRs):
            LR_stack[:,:,(i*3):(i*3)+3] = imread(img)/256 #256,256,Segments
            
        HR_stack = imread(HR[0])/256 #256,256,Segments

        X[image_number,:,:,:] = LR_stack
        Y[image_number,:,:,:] = HR_stack

    return X, Y



def dataset_stack_1D(dataset, segments):
    X = np.empty((len(dataset) * segments,256,256,3))
    Y = np.empty((len(dataset) * segments,256,256,3))

    image_count = 0

    for folder_path in tqdm(dataset):
        LRs = sorted(glob(folder_path+"/LR*.png")) #4 LRs
        HR = sorted(glob(folder_path+"/HR*.png")) #1 HR input 4 times


        for img in LRs:
            X[image_count,:,:,:] = imread(img)/256 #256,256,4
            Y[image_count,:,:,:] = imread(HR[0])/256 #256,256,4

            image_count += 1

    return X, Y


def save_stack(stack_save_dir, file_name, FourD_Stack):

    if not os.path.isdir(stack_save_dir):
        os.mkdir(stack_save_dir)

    np.save(os.path.join(stack_save_dir, file_name), FourD_Stack)
