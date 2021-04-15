#!/usr/bin/env python3

import glob, os
import cv2
IMG_SIZE = 600

def resize_image(image_path):
    
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
    cv2.imwrite('resized_' + image_path, resized_img)

def main():
    
    images = glob.glob('*.png') + glob.glob('*.jpg')

    for image_path in images:
        resize_image(image_path)

if __name__ == "__main__":
    main()