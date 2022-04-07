import os
from glob import glob
import shutil

# getting the filename from the user
file_path = sorted(glob("image_datasets/imagedata*"))
file_path.append("image_stacks")
file_path.append("MFUNet_trained500")
file_path.append("logs")

for folder in file_path:
    # checking whether file exists or not
    try:
       shutil.rmtree(folder)

    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
