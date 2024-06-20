import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np  
import time
import cv2   

from numba import cuda, njit
import math

def make_binary(img, kernel, binary_type):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    _,imb = cv2.threshold(img,0,255,binary_type+cv2.THRESH_OTSU)
    imb = cv2.morphologyEx(imb, cv2.MORPH_CLOSE, kernel)
    return imb

class twoPassComponentsDetect():
    def __init__(self,img, binary_type = None, ):
        t0 = time.time()
        self.connection_table = [0]
        self.kernel = np.ones((3, 3), np.uint8)
        self.label = np.zeros(img.shape[:2],dtype=np.int)
        self.row , self.col = img.shape[:2]
        if binary_type is None:
            imb = img
        else:
            imb = make_binary(img, binary_type)
        self.imb = cv2.copyMakeBorder(imb,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
        t1 = time.time()
        self.first_pass_opt()
        t2 = time.time()
        self.second_pass()
        t3 = time.time()
        print(t1-t0,t2-t1,t3-t2)

    def __call__(self, ):
        return self.num_label, self.label

    def first_pass_opt(self):
        label = 1
        for i in range(1,self.row + 1):
            for j in range(1, self.col + 1):
                if self.imb[i][j] == 0:
                    continue
                if self.imb[i-1][j] >0:
                    root = self.union_find(self.label[i-2][j-1])
                else:
                    if self.imb[i-1][j-1] >0:
                        if self.imb[i-1][j+1] >0:
                            root = self.union_find(self.label[i-2][j-2],self.label[i-2][j])
                        else:
                            root = self.union_find(self.label[i-2][j-2])
                    else:
                        if self.imb[i-1][j+1] >0:
                            if self.imb[i][j-1] >0:
                                root = self.union_find(self.label[i-1][j-2],self.label[i-2][j])
                            else:
                                root = self.union_find(self.label[i-2][j])
                        else:
                            if self.imb[i][j-1] >0:
                                root = self.union_find(self.label[i-1][j-2])
                            else:
                                self.connection_table.append(label)
                                root = label
                                label += 1
                self.label[i-1][j-1] = root
        self.flattenL()
        
    def second_pass(self):
        for i in range(self.row):
            for j in range(self.col):
                self.label[i][j] = self.connection_table[self.label[i][j]]
        return map 
                

    def get_root(self, i: int) -> int:
        root = i
        while self.connection_table[root] < root:
            root = self.connection_table[root]
        return root

    def set_root(self, i: int ,root: int) -> int:
        while self. connection_table[i] < i:
            j = self.connection_table[i]
            self.connection_table[i] = root
            i = j
        self.connection_table[i] = root

    def union_find(self, i: int,j: int=None) -> int:
        root = self.get_root(i)
        if j is not None and i != j:
            rootj = self.get_root(j)
            if root > rootj:
                root = rootj
            self.set_root(j,root)
        self.set_root(i,root)
        return root 
    
    def flattenL(self):
        k = 0
        for i in range(len(self.connection_table)):
            if self.connection_table[i] < i:
                j = self.connection_table[i] 
                self.connection_table[i] = self.connection_table[j]
            else:
                self.connection_table[i] = k
                k +=1
        self.num_label = k

def draw(map, num_label):
    map = map*180/num_label
    h,w = map.shape
    for i in range(h):
        for j in range(w):
            if map[i][j] > 0:
                map[i][j] += 60
    return np.round(map).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("2pass_test.png",0)
    _,imb = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k1 = np.ones((3, 3), np.uint8)
    imb = cv2.morphologyEx(imb, cv2.MORPH_CLOSE, k1)
    st = time.time()
    fun = twoPassComponentsDetect(imb)
    my_num,my_labels = fun()
    fn = time.time()
    print("time:", fn-st, "num_label:",my_num)
    result = draw(my_labels,my_num)
    st_cv = time.time()
    num,labels= cv2.connectedComponentsWithAlgorithm(imb,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_WU)
    print("num_label:",num)
    fn_cv = time.time()
    print("time:", fn_cv-st_cv, "num_label:",num)
    plt.imshow(result)
    plt.show()
    cv2.imwrite('a_new.jpg',result)        