import glob, os
import numpy as np
import cv2
import argparse


def create_flip_image(path_to_folder):
    for filename in glob.glob(os.path.join(path_to_folder, "*")):
        img = cv2.imread(filename)
        _, only_filename = os.path.split(filename)
        
        print(path_to_folder, only_filename) 
        flip_image = cv2.flip(img, 1)
        # cv2.imshow(only_filename, flip_image)
        # cv2.waitKey(0)
        
        print(path_to_folder + '/flip_' + only_filename) 
        cv2.imwrite(path_to_folder + '/flip_' + only_filename, flip_image)


def xywh2xyxy(x, y, w, h, image_width, image_height):
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data augmentation')
    parser.add_argument('-s', '--source', type=str, default='../images/Train/neg', help='source to folder images') 
    
    args = parser.parse_args()

    create_flip_image(args.source)