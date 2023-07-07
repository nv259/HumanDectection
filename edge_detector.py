import cv2
import numpy as np


def show_img(img):
    print(img)
    
    img = np.expand_dims(img, axis=2).astype(np.uint8)
    img = cv2.resize(img, [1000, 1000])

    cv2.imshow('origin', img)
    cv2.waitKey(0)
    

def conv(img, filter):
    padded_img = np.pad(img, pad_width=(1, 1), mode='constant', constant_values=0)
    
    """Apply filter"""
    rows, cols = padded_img.shape
    filter_size = filter.shape[0]
    
    result = np.zeros((rows - 2, cols - 2))
    
    for i in range(rows - filter_size + 1):
        for j in range(cols - filter_size + 1):
            sub_img = padded_img[i:i + filter_size, j:j+filter_size]
            
            filtered_value = np.sum(sub_img * filter)
            result[i, j] = filtered_value

    return result


if __name__ == '__main__':
    img = np.array([[210, 210, 220, 210, 220],
                    [220, 220, 180, 170, 150],
                    [210, 220, 50, 60, 60],
                    [70, 60, 60, 80, 90],
                    [60, 50, 70, 80, 60]])

    # Filters
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    
    vertical_edge = conv(img, Gx)
    horizontal_edge = conv(img, Gy) 
    G = np.zeros(vertical_edge.shape)
    
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i][j] = (vertical_edge[i][j]**2 + horizontal_edge[i][j]**2) ** (1/2) 

    # Normalize
    G_max = G.max()
    G = G / G_max * 255
    
    # print(vertical_edge)
    print(horizontal_edge)
    # print(G)
    # show_img(img)
