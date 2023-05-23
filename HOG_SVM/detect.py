import numpy as np
import cv2, joblib, argparse
from utils.plots import plot_one_box
from skimage.transform import pyramid_gaussian
from skimage.feature import hog


def sliding_windows(img, window_size, stride):
    for y in range(0, img.shape[0], stride[1]):
        for x in range(0, img.shape[1], stride[0]):
            yield (x, y, img[y : y + window_size[1], x : x + window_size[0]])


def main(source, model, output, isView, stride, width, height, downscale):
    # Initialize 
    img = cv2.imread(source)
    img = cv2.resize(img, [width, height])
    model = joblib.load(model)
    stride = [stride, stride] # Step size
    spatial_window_shape = [64, 128] # size of sliding window
    bounding_boxes = []
    
    # Detect
    for im in pyramid_gaussian(img, downscale=downscale):
        # If the scaled image is smaller than the spatial window, stop
        if im.shape[0] < spatial_window_shape[1] or im.shape[1] < spatial_window_shape[0]:
            break
        
        for (x, y, spatial_window) in sliding_windows(im, spatial_window_shape, stride):
            # If spatial window is flow out the boundary of image, skip
            if spatial_window.shape[0] != spatial_window_shape[1] or spatial_window.shape[1] != spatial_window_shape[0]:
                continue
            
            # Calculate hog of spatial window
            spatial_window = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_spatial_window = hog(spatial_window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
            hog_spatial_window = hog_spatial_window.reshape(1, -1)
            
            # Predict
            y_pred = model.predict(hog_spatial_window)
            if y_pred == 1 and model.decision_function(hog_spatial_window) > 0.5:
                # Append new bounding box
                pass 
        
    # Non-max-suppression
    
    # Draw bounding boxes

    # Show image
    if isView == True:
        cv2.imshow(output, img)
    

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Human Detection Using Linear SVM + HOG')
    parser.add_argument('-s', '--source', type=str, help='path of input file', required=True)
    parser.add_argument('-m', '--model', type=str, default='./models/LinearSVC.dat', help='path of trained model')
    parser.add_argument('-o', '--output', default='output.jpg', help='output file', required=True)
    parser.add_argument('-v', '--view', action='store_true', help='view output')
    parser.add_argument('--stride', type=int, default=9, help='step size')
    parser.add_argument('-w', '--width', type=int, default=400, help='image width')
    parser.add_argument('-h', '--height', type=int, default=256, help='image height')
    parser.add_argument('--downscale', type=float, default=1.25, help='downscale parameter for create image pyramid')
    
    args = parser.parse_args()

    main(args.source, args.model, args.output, args.view, args.stride, args.width, args.height)