#!/usr/bin/env python3

import glob
import os
import numpy as np

REL = os.getcwd()
INFORMATIVE_1 = REL + '/data/Training_data/Informative'
INFORMATIVE_2 = REL + '/data/Testing_data/Informative'
NON_INFORMATIVE_1 = REL + '/data/Testing_data/Non-Informative'
NON_INFORMATIVE_2 = REL + '/data/Training_data/Non-Informative'

def get_images():
    """
    returns informative and non-informative images with .png and .jpg extension 
    
    """

    informative = glob.glob(INFORMATIVE_1+'/*.png') + glob.glob(INFORMATIVE_2+'/*.png') + glob.glob(INFORMATIVE_1+'/*.jpg') + glob.glob(INFORMATIVE_2+'/*.jpg')

    non_informative = glob.glob(NON_INFORMATIVE_1+'/*.png') + glob.glob(NON_INFORMATIVE_2+'/*.png') + glob.glob(NON_INFORMATIVE_1+'/*.jpg') + glob.glob(NON_INFORMATIVE_2+'/*.jpg')
    
    return informative, non_informative
    

def rename_images(images, label):
    """
    attaches label _0 if class in informative or _1 if class in non-informative
    :params: images and their class
    
    """
    if label == 'informative':
        prefix = '_0'
    else:
        prefix = '_1'

    for image in images:
        name = image.split('/')[-1]
        path = '/'.join(image.split('/')[:-1])
        new_name = os.path.join(path, (name[:-4]+prefix+name[-4:]))
        print(image)
        print(new_name)
#         os.rename(image, new_name)
    
    return

if __name__ == "__main__":
    
    informative, non_informative = get_images()
    rename_images(informative, "informative")
    rename_images(non_informative, "non_informative")