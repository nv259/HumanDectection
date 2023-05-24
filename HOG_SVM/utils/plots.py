import random
import cv2
from PIL import Image, ImageDraw
import numpy as np


def plot_one_box(img, bb, color=None, line_thickness=3, cls='Human'):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
    color = color or [random.randint(0, 255) for _ in range(3)]
    upper_left, lower_right = (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3]))
    
    cv2.rectangle(img, upper_left, lower_right, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.putText(img, cls, (int(bb[0] - 2), int(bb[1] - 2)), 1, 0.75, (255, 255, 0), 1)

    return img


def plot_one_box_PIL(img, bb, color=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    
    draw.rectangle(bb, width=line_thickness, outline=tuple(color))  # plot
    return np.asarray(img)
    
    