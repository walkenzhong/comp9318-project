import pickle
import time
import scipy.spatial
import numpy as np
import copy


def pq(data, P, init_centroids, max_iter):
    # split_data,,将input数据的维度平均切割成p份
    dimensionality = len(data[0])
    dimensionality_after_split = int(dimensionality / P)
    data_after_split = []
    flag = 0
    data = np.array(data)
    while P > 0:
        split_data_list = data[:, flag:flag + dimensionality_after_split]
        flag += dimensionality_after_split
        P -= 1
        data_after_split.append(split_data_list)
    codebooks = []
    k = len(init_centroids)  # k是中心点的数量，即表示有多少个分类
    # 开始对每一个分割后的数据集进行处理
    for i in range(len(data_after_split)):
        # 复制初始的中心点
        tem_center = copy.deepcopy(init_centroids[i])
        # kmeans的循环最大次数为max_iter
        tem_max_iter = max_iter
        while tem_max_iter > 0:
            tem_max_iter -= 1
            # 用来存储每个中心点包含的vectors，由于中心点在变化，所以每次都需要清空
            center_cluster = []
            for num in range(len(tem_center)):
                center_cluster.append([])
            # 获取每个点与所有center之间的L1距离
            distance_list = scipy.spatial.distance.cdist(data_after_split[i], tem_center, metric='cityblock')
            # 获取每个vector对应的center的index，该index对应的是center_list中的index
            min_distance_index = np.argmin(distance_list, axis=1)
            # 将vector分配到其所属的center中
            for num in range(len(data_after_split[i])):
                center_point_index = min_distance_index[num]  # 获取该点对应的中心点index
                vector = data_after_split[i][num]  # 获取该点
                center_cluster[center_point_index].append(vector)  # 将该点加入中心点对应的cluster
            # 重新计算中心点
            old_center = copy.deepcopy(tem_center)  # 复制中心点用于比较
            for num in range(len(center_cluster)):
                # 如果该中心点cluster有对应的点加入，重新计算中心点
                if len(center_cluster[num]) > 0:
                    tem_center[num] = np.median(center_cluster[num], axis=0)
                else:
                    # 没有对应的点加入，中心点保持不变
                    continue
            # 与上一次循环的中心点进行比较，若完全相同，可提前跳出循环
            if ((old_center == tem_center).all()):
                break
            else:
                continue
        # 每一轮kmeans结束后，将得到的中心点及对应的code将入结果集
        codebooks.append(tem_center)
    # 生成codes
    codes = []
    for i in range(len(data_after_split)):
        # 计算与每个点的距离
        distance_list = scipy.spatial.distance.cdist(data_after_split[i], codebooks[i], metric='cityblock')
        min_distance_index = np.argmin(distance_list, axis=1)
        # 初始化codes
        tem_codes = np.linspace(-1, -1, len(data_after_split[i]))
        for num in range(len(data_after_split[i])):
            center_point_index = min_distance_index[num]
            tem_codes[num] = center_point_index  # 设置该点对应中心点的index
        codes.append(tem_codes)
    codebooks = np.array(codebooks, dtype='float32')
    codes = np.array(codes, dtype='uint8')
    # 将矩阵进行旋转
    codes = np.transpose(codes)
    return codebooks, codes


def query(queries, codebooks, codes, T):
    pass