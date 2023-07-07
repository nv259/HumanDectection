import numpy as np



if __name__=='__main__':
    gradient_direction = np.array([[80, 10],
                                   [165, 91]])
    gradient_magnitude = np.array([[2, 4],
                                   [85, 34]])
    
    hog = dict()
    for bin in range(0, 191, 20):
        hog[bin] = 0.0
        
    for i in range(gradient_direction.shape[0]):
        for j in range(gradient_magnitude.shape[1]):
            check = False
            x1, x0 = 0, 0 
            
            for k in hog.keys():
                if gradient_direction[i][j] == k:
                    hog[k] += gradient_magnitude[i][j]
                    check = True
                    break
                
                if gradient_direction[i][j] < k:
                    x1 = k
                    x0 = k - 20
                    check = False
                    break
            
            if check: 
                continue
            
            x = gradient_direction[i][j]
            y = gradient_magnitude[i][j]

            hog[x0] += y * (x1 - x) / (x1 - x0)
            hog[x1] += y * (x - x0) / (x1 - x0)
            
    hog[0] += hog[180]
    
    print(hog)