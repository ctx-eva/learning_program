import matplotlib.pyplot as plt 
import time
import numpy as np
import cv2

if __name__=='__main__':
    img = cv2.imread("2pass_test.png")
    _,imb = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k1 = np.ones((3, 3), np.uint8)
    imb = cv2.morphologyEx(imb, cv2.MORPH_CLOSE, k1)
    st_cv = time.time()
    nums, labels = cv2.connectedComponents(imb,connectivity=4)
    #can use cv2.connectedComponentsWithAlgorithm select Algorithm with different neighbourhood
    #the default Algorithm in cv2.connectedComponents is BBDT for 8 connectivity, SAUF algorithm for 4 connectivity
    fn_cv = time.time()
    print(fn_cv-st_cv)
    map = np.zeros(imb.shape)
    for label in labels:
        map += label * 180/nums +60
    plt.imshow(map)
    plt.show()