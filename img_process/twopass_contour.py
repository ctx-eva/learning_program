from heapq import merge
import numpy as np  
from pathlib import Path
import cv2   
import matplotlib.pyplot as plt
import time
from numba import cuda, njit
import math

#====================================
#Reference: https://www.jianshu.com/p/faba96cb624a , https://blog.csdn.net/icvpr/article/details/10259577
#====================================

#由于循环方向原因对于邻域范围没有必要进行全搜索
OFFSETS_4 = np.array([[0, -1], [-1, 0]]) # 左，上
OFFSETS_8 = np.array([[0, -1], [-1,  0],[-1, -1], [-1,  1]]) # 左,上,左上,右上

def first_pass(imb,k):
    '''
    第一次扫描：
    1>访问当前像素B(x,y),如果B(x,y) == 1:
    a、如果B(x,y)的领域中标签值都为0,则赋予B(x,y)一个label,label += 1;
    b、如果B(x,y)的领域中有标签值 > 0的像素Neighbors, 将Neighbors中的最小值赋予给B(x,y)
    2>记录Neighbors中各个值(label)之间的相等关系,即这些值(label)同属同一个连通区域.
    first_pass有搜索方向不能用gpu加速
    '''
    label_map = np.zeros(imb.shape)
    h,w = imb.shape
    label = 1
    label_dict = {}  #label关系字典:key当前关系中的最小值,value当前关系关联的所有值
    for i in range(h):
        for j in range(w):
            if imb[i][j] == 0:
                continue
            else:
                near = []
                for p in k:
                    p_ = [i+p[0],j+p[1]]
                    if p_[0] < 0 or p_[1] <0 or p_[1] >=w:  #判断邻域越界
                        continue
                    else:
                        if label_map[p_[0]][p_[1]] >0:
                            near.append(label_map[p_[0]][p_[1]])  #记录邻域大于0的标签值
                if len(near):
                    minval = min(near)  #将邻域最小值赋予给B(i,j)
                    label_map[i][j] = minval
                    near_set = set(near)
                    if len(near_set) > 1:  #如果邻域出现多个不为0的标签值
                        merge = False
                        merge_list_v = near_set
                        merge_list_k = []
                        for key,value in label_dict.items():
                            if len(near_set & value):  #判断邻域和所有的label关系是否有交集
                                merge = True
                                merge_list_v |= value  #有交集则求二者的并集
                                merge_list_k.append(key)
                        if merge:
                            if len(merge_list_k) > 1:
                                for mk in merge_list_k:
                                    del label_dict[mk]  #如果邻域和多个label关系同时存在交集,删除原有label关系的对应键值对
                            label_dict[min(merge_list_v)] = merge_list_v
                        else:
                            label_dict[minval] = near_set
                else:
                    label_map[i][j] = label  #标签值都为0,则赋予B(i,j)一个label,label += 1
                    label += 1
    return label_map, label_dict

# def second_pass_(map,label_list,min_list):
#     '''
#     列表生成式不能提前跳过循环,速度并不快
#     '''
#     h,w = map.shape
#     result = [[min_list[label_list.index(map[i][j])] if map[i][j] in label_list else map[i][j] for j in range(w)] for i in range(h)]
#     return np.array(result)          

# def second_pass1(map,label_list,min_list):
#     '''
#     比列表生成式还慢
#     '''
#     h,w = map.shape
#     for i in range(h):
#         for j in range(w):  
#             val = map[i][j]
#             if val in min_list or val == 0:
#                 continue
#             if val in label_list:
#                 map[i][j] = min_list[label_list.index(map[i][j])]

#     return map

# def two_pass_(img,kernel=OFFSETS_4):
#     label_map, label_dict = first_pass(img,k=kernel)    
#     label_list = []
#     label_list_min = []
#     for k,v in label_dict.items():
#         for i in v:
#             label_list.append(i)
#             label_list_min.append(k)
#     map = second_pass_(label_map,label_list,label_list_min)  
#     return map    

def second_pass(map,label_dict):
    '''
    第二次扫描：
    访问当前像素B(x,y),如果B(x,y) > 1, 找到与label = B(x,y)同属相等关系的一个最小label值,赋予给B(x,y)
    用continue 和 break 及早结束循环可以加快速度
    '''
    h,w = map.shape
    for i in range(h):
        for j in range(w):  
            val = map[i][j]   
            if val == 0 or val in label_dict.keys() :  #值是0或者相等关系的一个最小label值,值不变
                continue
            for k,v in label_dict.items():
                if val in v:
                    map[i][j] = k
                    break
    return map   

@njit
def get_index_nb(A, k):
    for i in range(len(A)):
        if A[i] == k:
            return i
    return -1

@njit
def get_in(A, k):
    for i in range(len(A)):
        if A[i] == k:
            return True
    return False

@cuda.jit
def second_pass_gpu(map,label_list,min_list):
    """
    使用numba加速second_pass
    """
    tx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y
    val = map[tx,ty]
    if val == 0:
        map[tx,ty] = val
    elif get_in(min_list,val):
        map[tx,ty] = val
    elif get_in(label_list,val):
        map[tx,ty] = min_list[get_index_nb(label_list,val)]

def two_pass(img,kernel=OFFSETS_4):
    t1 = time.time()
    label_map, label_dict = first_pass(img,k=kernel)
    t2 = time.time()
    map = second_pass(label_map,label_dict)
    t3 = time.time()
    print("pass1time",t2-t1)
    print("pass2time",t3-t2)
    return map   

def two_pass_gpu(img,kernel=OFFSETS_4):   
    gpu_st = time.time()
    label_map, label_dict = first_pass(img,k=kernel)
    dst_gpu = img.copy()
    label_list = []
    label_list_min = []
    for k,v in label_dict.items():
        for i in v:
            label_list.append(i)
            label_list_min.append(k)
    dImg = cuda.to_device(label_map)
    label_list = cuda.to_device(label_list)
    label_list_min = cuda.to_device(label_list_min)
    rows,cols = img.shape
    threadsperblock = (16,16)
    blockspergrid_x = int(math.ceil(rows/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(cols/threadsperblock[0]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    cuda.synchronize()
    second_pass_gpu[blockspergrid,threadsperblock](dImg,label_list,label_list_min)
    cuda.synchronize()
    dst_gpu = dImg.copy_to_host()
    gpu_end = time.time()
    print("pass2gpu",gpu_end-gpu_st)  
    return dst_gpu    

def draw(map):
    values = list(set(map.flatten().tolist()))
    h,w = map.shape
    for i in range(h):
        for j in range(w):
            if map[i][j] > 0:
                map[i][j] = values.index(map[i][j]) * 180/len(values) + 60
    return np.round(map).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("2pass_test.png",0)
    _,imb = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k1 = np.ones((3, 3), np.uint8)
    imb = cv2.morphologyEx(imb, cv2.MORPH_CLOSE, k1)
    h,w = imb.shape
    st = time.time()
    map = two_pass_gpu(imb)
    fn = time.time()
    print(fn-st)
    result = draw(map)
    plt.imshow(result)
    plt.show()
    cv2.imwrite('a_new.jpg',result)