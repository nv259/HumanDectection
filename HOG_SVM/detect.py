import random
import sys
sys.path.append('D:\\repo\\HumanDectection\\HOG_SVM')
sys.path.append('D:\\repo\\HumanDectection\\HOG_SVM\\yolov7')

# suppress warning
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2, joblib, argparse, os, glob
from imutils.object_detection import non_max_suppression
from skimage import color
from utils.plots import plot_one_box
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
from utils.metrics import IoU
from yolov7.yolo_detect import YoloDetect


model = []


def sliding_windows(img, window_size, stride):
    for y in range(0, img.shape[0], stride[1]):
        for x in range(0, img.shape[1], stride[0]):
            yield (x, y, img[y : y + window_size[1], x : x + window_size[0]])


def xywh2xyxy(xywh, image_width, image_height):
    x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height
    
    return [x1, y1, x2, y2]


def do_hard_negative_mining(bbs, filename, origin_size, scaled_size):
    print('Attention!')
    for bb in bbs:
        # unscaled bounding box
        pick = bb.copy()
        pick[0] = pick[0] * (origin_size[0] / scaled_size[0])
        pick[1] = pick[1] * (origin_size[1] / scaled_size[1])
        pick[2] = pick[2] * (origin_size[0] / scaled_size[0])
        pick[3] = pick[3] * (origin_size[1] / scaled_size[1])
        intersect_count = 0

        # hard negative mining
        for xywh in sub_detector.detect(filename):
            if IoU(pick, xywh2xyxy(xywh, origin_size[0], origin_size[1])) > 0.05:
                intersect_count = intersect_count + 1

        if intersect_count == 0:
            img = cv2.imread(filename)
            img = img[pick[1] : pick[3], pick[0] : pick[2]]
            # cv2.imshow('false positive', img)
            # cv2.waitKey(0)
            cv2.imwrite("D:\\repo\\HumanDectection\\HOG_SVM\\images\\Train\\neg\\" + str(random.randint(111111, 999999)) + '.jpg', img)


def process_one_image(filename, width, height, spatial_window_shape, stride, downscale, output, is_view,
                      hard_negative_mining, overlap_thresh, conf):
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

    # Draw bounding boxes
    for bounding_box in bounding_boxes:
        plot_one_box(img, bounding_box)

    img = cv2.resize(img, origin_size)
    # Show image
    if is_view:
        cv2.imshow(output, img)
        cv2.waitKey(0)

    # hard negative mining
    if hard_negative_mining:
        do_hard_negative_mining(bounding_boxes, filename, origin_size, [width, height])
    else:
        cv2.imwrite('./run/' + os.path.split(filename)[1], img)
        print("Successfully saved in run/" + os.path.split(filename)[1])


def main(source, model_source, output, is_view, stride, width, height, downscale, hard_negative_mining, overlap_thresh, conf):
    print(f'Input file: {source}\nModel: {model_source}\nOutput file: {output}\nStep size: {stride}\n'
          f'(Image width, Image height): ({width}, {height})\nPyramid Gaussian downscale: {downscale}')

    global model

    model = joblib.load(model_source)
    spatial_window_shape = [64, 128]  # size of sliding window
    stride = [stride, stride]  # Step size

    if not (source.endswith(".jpg") or source.endswith(".png")):
        for filename in glob.glob(os.path.join(source, "*")):
            if hard_negative_mining:
                print(filename)

            process_one_image(filename, width, height, spatial_window_shape, stride, downscale, output, is_view,
                            hard_negative_mining, overlap_thresh, conf)
    else:
        process_one_image(source, width, height, spatial_window_shape, stride, downscale, output, is_view,
                          hard_negative_mining, overlap_thresh, conf)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Human Detection Using Linear SVM + HOG')
    parser.add_argument('-s', '--source', type=str, default='./test.png', help='path of input file')
    parser.add_argument('-m', '--model', type=str, default='./models/LinearSVC.dat', help='path of trained model')
    parser.add_argument('-o', '--output', default='output.jpg', help='output file')
    parser.add_argument('-v', '--view', action='store_true', help='view output')
    parser.add_argument('--stride', type=int, default=9, help='step size')
    parser.add_argument('-w', '--width', type=int, default=400, help='image width')
    parser.add_argument('-H', '--height', type=int, default=256, help='image height')
    parser.add_argument('--downscale', type=float, default=1.25, help='downscale parameter for creating image pyramid')
    parser.add_argument('--hard-negative-mining', action='store_true', help='enable hard negative mining (source must '
                                                                            'be a folder, not image)')
    parser.add_argument('-t', '--overlap_thresh', type=float, default=0.2, help='Overlap thresh for non-max suppression')
    parser.add_argument('-c', '--confidence-score', type=float, default=0.5, help='Threshold for confidence score')

    args = parser.parse_args()
    
    if args.hard_negative_mining:
        sub_detector = YoloDetect(weights='D:\\repo\\HumanDectection\\HOG_SVM\\yolov7\\trained__pt\\yolov7.pt')

    main(args.source, args.model, args.output, args.view, args.stride, args.width, args.height, args.downscale,
         args.hard_negative_mining, args.overlap_thresh, args.confidence_score)
