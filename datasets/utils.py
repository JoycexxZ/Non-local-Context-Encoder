import os
import numpy as np
from PIL import Image
import cv2

def get_filenames(path):
    names = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            fname, _ = os.path.splitext(filespath)
            names.append(fname)
    return names

def get_data_JPCL(data_path, mask_path, name):
    data_path = os.path.join(data_path, name+".IMG")
    shape = (2048, 2048, 1)
    dtype = np.dtype('>u2') 
    fid = open(data_path, 'rb')
    data = np.fromfile(fid, dtype)
    image = data.reshape(shape)

    mask_left_path = os.path.join(mask_path, "left lung", name+".gif")
    mask_right_path = os.path.join(mask_path, "right lung", name+".gif")
    mask_left = np.array(Image.open(mask_left_path))
    mask_right = np.array(Image.open(mask_right_path))
    mask = np.minimum(mask_left + mask_right, 1)

    return image, mask
    
def get_data_ISBI(data_path, mask_path, name):
    data_path = os.path.join(data_path, name+".jpg")
    image = Image.open(data_path)
    mask_path = os.path.join(mask_path, name+"_Segmentation.png")
    mask = Image.open(mask_path)
    mask = np.minimum(mask, 1)
    mask = Image.fromarray(mask) 

    return image, mask

def reshape_image(image, image_size):
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape

    if h > w:
        image = image[w//2:w//2+w, :]
    elif h < w:
        image = image[:, h//2:h//2+h]

    image = cv2.resize(image, (image_size, image_size))
    return image

def norm_JPCL(image):
    return image/4096