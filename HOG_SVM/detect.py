import numpy as np
import cv2, joblib, argparse


def sliding_windows(img, window_size, stride):
    for y in range(0, img.shape[0], stride[1]):
        for x in range(0, img.shape[1], stride[0]):
            yield (x, y, img[y : y + window_size[1], x : x + window_size[0]])


def main(source, model, output, isView):
    # Initialize 
    img = cv2.imread(source)
    img = cv2.resize(img, [])
    model = joblib.load(model)
    stride = [9, 9] # Step size
    window_size = [64, 128] # Train img size


    
    

if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description='Human Detection Using Linear SVM + HOG')
    parser.add_argument('-s', '--source', type=str, help='path of input file', required=True)
    parser.add_argument('-m', '--model', type=str, default='./models/LinearSVC.dat', help='path of trained model')
    parser.add_argument('-o', '--output', default='output.jpg', help='output file', required=True)
    parser.add_argument('-v', '--view', action='store_true', help='view output')
    
    args = parser.parse_args()

    main(args.source, args.model, args.output, args.view)