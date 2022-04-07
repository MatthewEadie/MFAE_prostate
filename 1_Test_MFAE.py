"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from cv2 import imwrite
from math import floor

#----------
# Settings
model_fp = "./MFUNet_trained500"
test_index = 3 #index of image to test model on
path_datasets = "image_stacks"
save_path = "SR_4DImages"
#----------



load_datasets = True
test_model = False
reconstruct_dataset = True

if(load_datasets):
    X_test = np.load(os.path.join(path_datasets, "X_test4D.npy")) #(32,256,256,4) Images to run through model
    Y_test = np.load(os.path.join(path_datasets, "Y_test4D.npy")) #(32,256,256,4) HR images for comparison

    MF_UNet = load_model(model_fp,compile=False)
    MF_UNet.compile(optimizer='adam',loss='mse')

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of Y_test: {Y_test.shape}")



if(test_model):
    test_image = X_test[test_index:test_index+1]

    print(f"Shape of test_image: {test_image.shape}")

    X_pred = MF_UNet(test_image)

    fig, ax = plt.subplots(3)
    ax[0].imshow(X_test[test_index,:,:,0])
    ax[0].set_title('LR')

    ax[1].imshow(X_pred[0,:,:,0])
    ax[1].set_title('Prediction')

    ax[2].imshow(Y_test[test_index])
    ax[2].set_title('HR')
    
    plt.show()



import time

print("hello")


start = time.time()

if(reconstruct_dataset):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for index in range(floor(X_test.shape[0])):
        test_image = X_test[index:index+1,]
        X_pred = MF_UNet(test_image)



        plt.imsave(save_path + "/SR{}.png".format(index), X_pred[0,:,:,0], cmap='gray')

end = time.time()
print(f'Time taken: {end - start}seconds')