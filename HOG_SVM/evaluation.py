import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from utils.metrics import IoU
import joblib
import cv2
from skimage.feature import hog 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import time 
import re


model = joblib.load('./models/LinearSVC.dat')
eval_images = './eval/images'
eval_annotations = './eval/annotations'


def get_annotations(filename):
    gt_boxes = []
    with open(os.path.join(eval_annotations, filename)) as f:
        lines = f.readlines()
        
        for line in lines:
            if 'Bounding box for object' in line: 
                gt_boxes.append(list(map(int, re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[1:])))
                print(gt_boxes) 

    return np.array(gt_boxes)


def sliding_windows(img, window_size, stride):
    for y in range(0, img.shape[0], stride[1]):
        for x in range(0, img.shape[1], stride[0]):
            yield (x, y, img[y : y + window_size[1], x : x + window_size[0]])


def process_one_image(filename, width=400, height=256, spatial_window_shape=[64, 128], stride=[9, 9], downscale=1.25
                      , overlap_thresh=0.2, conf=1.2):
    # Initialize
    global model

    img = cv2.imread(filename)
    origin_size = [img.shape[1], img.shape[0]]
    img = cv2.resize(img, [width, height])
    bounding_boxes = []
    scores = []
    curr_scale = 0.8

    # Detect
    for im in pyramid_gaussian(img, downscale=downscale):
        # If the scaled image is smaller than the spatial window, stop
        if im.shape[0] < spatial_window_shape[1] or im.shape[1] < spatial_window_shape[0]:
            break

        curr_scale = downscale * curr_scale

        for (x, y, spatial_window) in sliding_windows(im, spatial_window_shape, stride):
            # If spatial window is flow out the boundary of image, skip
            if spatial_window.shape[0] != spatial_window_shape[1] or spatial_window.shape[1] != spatial_window_shape[0]:
                continue

            # Calculate hog of spatial window
            spatial_window = spatial_window * 255
            spatial_window = spatial_window.astype(np.uint8)
            spatial_window = cv2.cvtColor(spatial_window, cv2.COLOR_BGR2GRAY)
            # spatial_window = color.rgb2gray(spatial_window)
            hog_spatial_window = hog(spatial_window, orientations=9, pixels_per_cell=(8, 8),
                                     visualize=False, cells_per_block=(3, 3))
            hog_spatial_window = hog_spatial_window.reshape(1, -1)

            # Predict
            y_pred = model.predict(hog_spatial_window)
            if y_pred == 1:
                score = model.decision_function(hog_spatial_window)
                # print(score)
                # print('human')
                if score > conf:
                    # Append new bounding box
                    # print('surely human')

                    bb_x, bb_y = int(curr_scale * x), int(curr_scale * y)
                    bb_w, bb_h = int(curr_scale * spatial_window_shape[0]), int(curr_scale * spatial_window_shape[1])

                    bounding_boxes.append((bb_x, bb_y, bb_x + bb_w, bb_y + bb_h))
                    scores.append(score[0])

    # Non-max-suppression
    bounding_boxes = np.array(bounding_boxes)
    scores = np.array(scores)
    bounding_boxes = non_max_suppression(bounding_boxes, probs=scores, overlapThresh=overlap_thresh)

    # Return coordinates in origin image
    for bb in bounding_boxes:
        bb[0] = bb[0] * (origin_size[0] / width)
        bb[1] = bb[1] * (origin_size[1] / height)
        bb[2] = bb[2] * (origin_size[0] / width)
        bb[3] = bb[3] * (origin_size[1] / height)
        
    return bounding_boxes


def main():
    for img in glob.glob(os.path.join(eval_images, "*")):
        # Get bounding boxes and ground truth boxes
        print(img)
        bounding_boxes = process_one_image(img)
        print('#bbes:', bounding_boxes.shape)
        gt_boxes = get_annotations(os.path.split(img)[1][:-4] + '.txt')
        
        # Count True Positive
        true_positive = 0  
        for bounding_box in bounding_boxes:
            count = 0
            for gt_box in gt_boxes:
                if IoU(bounding_box, gt_box) >= 0.5:
                    count = count + 1
            if count != 0:
                true_positive = true_positive + 1
        
        # Precision
        precision = 0 
        
        # Recall 
        recall = 0
        
        # F1-score
        f1_score = 2 * (precision * recall) / (precision + recall)
        

if __name__ == '__main__':
    main()