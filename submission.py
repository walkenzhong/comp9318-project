import pickle
import time
import scipy.spatial
import numpy as np
import copy
#codes结果有问题
def pq(data, P, init_centroids, max_iter):
    #split_data
    dimensionality = len(data[0])
    dimensionality_after_split = int(dimensionality/P)
    data_after_split = []
    flag = 0
    data = np.array(data)
    while P > 0:
        #split_data_list = data[[0,len(data)-1]]
        split_data_list = data[:,flag:flag + dimensionality_after_split]
        flag += dimensionality_after_split
        P -= 1
        data_after_split.append(split_data_list)
    codebooks = []
    k = len(init_centroids)
    codes = []
    for i in range(len(data_after_split)):
        tem_center = init_centroids[i][:]
        tem_max_iter = max_iter
        while tem_max_iter > 0:
            tem_max_iter -= 1
            #初始化一个用于存放每个点对应center的list
            tem_codes = np.linspace(-1,-1,len(data_after_split[i]))
            #用来存储每个中心点包含的vectors，由于中心点在变化，所以每次都需要清空
            center_cluster = []
            for num in range(len(tem_center)):
                center_cluster.append([])
            #获取每个点与所有center之间的距离
            distance_list = scipy.spatial.distance.cdist(data_after_split[i], tem_center, metric='cityblock')
            #获取每个vector对应的center的index，该index对应的是center_list中的index
            min_distance_index = np.argmin(distance_list,axis=1)
            #将vector分配到其所属的center中
            for num in range(len(data_after_split[i])):
                center_point_index = min_distance_index[num]
                vector = data_after_split[i][num]
                tem_codes[num] = center_point_index
                center_cluster[center_point_index].append(vector)
            #重新计算中心点
            old_center = copy.deepcopy(tem_center)
            for num in range(len(center_cluster)):
                if len(center_cluster[num]) > 0:
                    tem_center[num] = np.median(center_cluster[num],axis=0)
                    #print(f'update{tem_center[num]}')
                else:
                    continue
            #当中心点与上一次完全一致
            if((old_center == tem_center).all()):
                #print(old_center == tem_center)
                #print('all right')
                break
            else:
                #print('not same')
                continue
        codebooks.append(tem_center)
        codes.append(tem_codes)
    codebooks = np.array(codebooks,dtype = 'float32')
    codes = np.array(codes,dtype = 'uint8')
    codes = np.transpose(codes)
    return codebooks, codes
def query(queries, codebooks, codes, T):
    pass