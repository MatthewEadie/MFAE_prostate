"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

import cv2
import utils.preprocessing as utils
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from math import floor
from numpy import zeros
from numpy.core.function_base import linspace
import math


#----------
# Settings

dataset_amount = 15 #Number of datasets
dataset_dir = "image_datasets" #Location of image datasets
dataset_output_dir = "image_stacks"
threshold_scale = 1 #Change the scale of mask
mask_name = "thresh256_x" + str(threshold_scale)
originals_dir = dataset_dir + "/originals/"
resize_dir = dataset_dir + "/originals_resized/"

img_resolution = 0.316
bundle_resolution = 0.32 #um/pixel

rot_left = -10
rot_right = 10
rot_segments = 10

create_rotations = False
Resize_Images = False
Slice_Originals = False
Create_Overlays = False
Create_4DStack = True
Create_1DStack = False
#----------


if(create_rotations):
    thresh300 = Image.open(dataset_dir + "/masks/" + "thresh300.png") #Need to load image with PIL

    Mask256 = zeros((256,256))

    for rot in linspace(rot_left, rot_right, rot_segments): #-10,10,11
        bundle_rot = thresh300.rotate(rot)
        
        height, width  = bundle_rot.size

        center = [math.floor(height/2), math.floor(width/2)]

        top = center[0] - 128
        bottom = center[0] + 128
        left = center[1] - 128
        right = center[1] + 128

        Mask256 = bundle_rot.crop((left,top,right,bottom))

        #Mask256.show("MAsk")

        Mask256.save(dataset_dir + "/masks/" + f'Mask256_{math.floor(rot)}.png', "PNG")


if(Resize_Images):

    image_names= os.listdir(originals_dir)

    scale_factor = img_resolution / bundle_resolution  

    for image_name in tqdm(image_names):
        img = cv2.imread(originals_dir + image_name)

        Xsize = floor(img.shape[1] * scale_factor)
        Ysize = floor(img.shape[0] * scale_factor)

        img_resized = utils.resize_image(img, Xsize, Ysize)

        if not os.path.isdir(resize_dir):
            os.mkdir(resize_dir)


        cv2.imwrite(resize_dir + image_name, img_resized)



if(Slice_Originals):
    dataset_count = 0

    Train_images = sorted(glob(originals_dir + "Image*"))
    Test_images = sorted(glob(originals_dir + "TestImage*"))

    print("Creating training slices from originals")
    for image_path in tqdm(Train_images):
    #Load in original images
    #image_path = Train_images[0]
        dataset_image = utils.load_original(image_path)

        #divide originals into 256x256 greyscale slices
        #save slices on original image into new folder per slice 
        dataset_count = utils.div_original(dataset_dir, dataset_count, dataset_image, train_set = True)

    print("Creating testing slices from test originals")
    for image_path in tqdm(Test_images):
    #Load in original images
    #image_path = Test_images[0]
        dataset_image = utils.load_original(image_path)

        dataset_count = utils.div_original(dataset_dir, dataset_count, dataset_image, train_set = False)

    
    

if(Create_Overlays):
    print("Creating slice overlays")

    datasets = sorted(glob(dataset_dir +"/imagedata*"))
    masks = sorted(glob(dataset_dir +"/masks/Mask256*"))

    for folder_path in tqdm(datasets):
        #folder_path = datasets[0]

        img_HR = cv2.imread(folder_path + "/HR.png")/256



        for c,mask_path in enumerate(masks):
            mask = cv2.imread(mask_path)

            masked = img_HR * mask

            bundle_ave = zeros((masked.shape[1], masked.shape[0], masked.shape[2]))

            for channels in range(mask.shape[2]):
                bundle_ave[:,:,channels] = utils.AverageCores(masked[:,:,channels])

            cv2.imwrite(folder_path + f'/LR{c}.png', bundle_ave)




if (Create_4DStack):

    X_train = []; Y_train = []
    X_test = []; Y_test = []
    X_val = []; Y_val = []

    #training_datasets = sorted(glob(dataset_dir +"/imagedata_training*"))
    testing_datasets = sorted(glob(dataset_dir +"/imagedata_testing*"))
    testing_datasets.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    #print(testing_datasets)

    #print("Creating training 4D stacks")
    #X_train_stack, Y_train_stack = utils.dataset_stack_4D_colour(training_datasets, rot_segments)

    print("Creating testing 4D stacks")
    X_test_stack, Y_test_stack = utils.dataset_stack_4D_colour(testing_datasets, rot_segments)




    ##Divide training datasets into training and validation
    #dataset_no = X_train_stack.shape[0]
    #validation_split = int((dataset_no / 10) * 2)
    #training_split = int(validation_split + 1)

    #X_val = X_train_stack[0:validation_split,:,:,:] #First 20% 
    #Y_val = Y_train_stack[0:validation_split] #First 20% only 3 channels

    #X_train = X_train_stack[training_split:,:,:,:] #Remaining 80%
    #Y_train = Y_train_stack[training_split:] #Remaining 80% 

    #print(f"X_train_stack shape: {X_train_stack.shape}")    #(88,256,256,4)
    #print(f"Y_train_stack shape: {Y_train_stack.shape}")    #(88,256,256,4)

    #print(f"X_test_stack shape: {X_test_stack.shape}")      #(32,256,256,4)
    #print(f"Y_test_stack shape: {Y_test_stack.shape}")      #(32,256,256,4)

    #utils.save_stack(dataset_output_dir, 'X_train4D.npy', X_train)
    #utils.save_stack(dataset_output_dir, 'Y_train4D.npy', Y_train)

    #utils.save_stack(dataset_output_dir, 'X_val4D.npy', X_val)
    #utils.save_stack(dataset_output_dir, 'Y_val4D.npy', Y_val)

    utils.save_stack(dataset_output_dir, 'X_test4D.npy', X_test_stack)
    utils.save_stack(dataset_output_dir, 'Y_test4D.npy', Y_test_stack)





if (Create_1DStack):

    X_train = []; Y_train = []
    X_test = []; Y_test = []

    training_datasets = sorted(glob(dataset_dir +"/imagedata_training*"))
    testing_datasets = sorted(glob(dataset_dir +"/imagedata_testing*"))

    print("Creating training 1D stacks")
    X_train_stack, Y_train_stack = utils.dataset_stack_1D(training_datasets, rot_segments)

    print("Creating testing 1D stacks")
    X_test_stack, Y_test_stack = utils.dataset_stack_1D(testing_datasets, rot_segments)

    #Divide training datasets into training and validation
    dataset_no = X_train_stack.shape[0]
    validation_split = int((dataset_no / 10) * 2)
    training_split = int(validation_split + 1)

    X_val = X_train_stack[0:validation_split,:,:,:] #First 20% 
    Y_val = Y_train_stack[0:validation_split] #First 20% only 3 channels

    X_train = X_train_stack[training_split:,:,:,:] #Remaining 80%
    Y_train = Y_train_stack[training_split:] #Remaining 80% 


    print(f"X_train_stack shape: {X_train_stack.shape}")    #(88,256,256,4)
    print(f"Y_train_stack shape: {Y_train_stack.shape}")    #(88,256,256,4)

    print(f"X_test_stack shape: {X_test_stack.shape}")      #(32,256,256,4)
    print(f"Y_test_stack shape: {Y_test_stack.shape}")      #(32,256,256,4)

    #cv2.imshow("X_val", X_val[0])
    #cv2.imshow("Y_val", Y_val[0])

    #cv2.imshow("X_train", X_train[0])
    #cv2.imshow("Y_train", Y_train[0])

    utils.save_stack(dataset_output_dir, 'X_train1D.npy', X_train)
    utils.save_stack(dataset_output_dir, 'Y_train1D.npy', Y_train)

    utils.save_stack(dataset_output_dir, 'X_val1D.npy', X_val)
    utils.save_stack(dataset_output_dir, 'Y_val1D.npy', Y_val)

    utils.save_stack(dataset_output_dir, 'X_test1D.npy', X_test_stack)
    utils.save_stack(dataset_output_dir, 'Y_test1D.npy', Y_test_stack)



cv2.waitKey(0)
