import submission
import pickle
import time
import scipy.spatial
import numpy as np
#data是一组多维的数据，P是维度会被分为几段，init_centroids是初始的中心点(代表点)，max_iter是kmeans最大的递归次数
#返回值为codebooks及codes，codebooks为所有代表点，code为data set与代表点的对应关系表

#input data=[[1,0]] center_set=[[0,1]] return 2.0
#输入data为含有一个或多个vector的list,center_set为一个或多个中心点的list，两个list中的column需要相同
def get_L1_distance(data,center_set):
    distance = scipy.spatial.distance.cdist(data, center_set, metric='cityblock')
    return distance
    
def kmeans(data,k,init_centroids,max_iter):
    #用来存储中心点的list
    center_list = init_centroids[:]
    while max_iter > 0:
        #max_iter用于停止kmeans算法
        max_iter -= 1
        #初始化一个用于存放每个点对应center的list
        vector_center_index_list = []
        for i in range(len(data)):
            vector_center_index_list.append(-1)
        #用来存储每个中心点包含的vectors，由于中心点在变化，所以每次都需要
        center_cluster = []
        for i in range(len(center_list)):
            center_cluster.append([])
        #获取每个点与所有center之间的距离
        distance_list = get_L1_distance(data,center_list)
        #获取每个vector对应的center的index，该index对应的是center_list中的index
        min_distance_index = np.argmin(distance_list,axis=1)
        #将vector分配到其所属的center中
        for i in range(len(data)):
            center_point_index = min_distance_index[i]
            vector = data[i]
            vector_center_index_list[i] = center_point_index
            center_cluster[center_point_index].append(vector)
        #重新计算中心点
        #1,清空原有中心点
        center_list = []
        #2,按照平均值，重新计算中心点
        for i in center_cluster:
            #print(i)
            new_center = np.median(i,axis=0)
            center_list.append(new_center)
    return center_list,vector_center_index_list
#把data的dimensionality划分为P份，返回为切割后的list set
def split_data(data,P):
    dimensionality = len(data[0])
    dimensionality_after_split = int(dimensionality/P)
    split_data_set = []
    flag = 0
    data = np.array(data)
    while P > 0:
        #split_data_list = data[[0,len(data)-1]]
        split_data_list = data[:,flag:flag + dimensionality_after_split]
        flag += dimensionality_after_split
        P -= 1
        split_data_set.append(split_data_list)
    return split_data_set
    
def pq(data, P, init_centroids, max_iter):
    '''
    print(len(data[0]))
    print(f'P:{P}')
    print(len(init_centroids[0][0]))
    print(init_centroids[1])
    print(f'max_iter:{max_iter}')
    distance = get_L1_distance([1,0],[0,1])
    print(distance)
    '''
    #根据P划分data
    data_after_split = split_data(data,P)
    #print(data_after_split)
    center_set = []
    k = len(init_centroids)
    codes = []
    for i in range(len(data_after_split)):
        tem_center,tem_codes = kmeans(data_after_split[i],k,init_centroids[i],max_iter)
        center_set.append(tem_center)
        codes.append(tem_codes)
    center_set = np.array(center_set)
    codes = np.array(codes)
    codes = np.transpose(codes)
    '''
    print(f'p:{P}')
    print(f'k:{len(init_centroids[0])}')
    print(f'M:{len(data)}')
    print(center_set)
    print(codes)
    print(np.shape(center_set))
    print(np.shape(codes))
    codebooks = []
    '''
    codebooks = center_set
    return codebooks, codes

def query(queries, codebooks, codes, T):
    print(queries)
    print(T)

    
if __name__ == '__main__':
    # How to run your implementation for Part 1
    with open('./toy_example/Data_File', 'rb') as f:
        Data_File = pickle.load(f, encoding = 'bytes')
    data = Data_File
    with open('./toy_example/Centroids_File', 'rb') as f:
        Centroids_File = pickle.load(f, encoding = 'bytes')
    centroids = Centroids_File
    start = time.time()
    codebooks, codes = pq(data, P=2, init_centroids=centroids, max_iter = 20)
    # How to run your implementation for Part 2
    with open('./toy_example/Query_File', 'rb') as f:
        Query_File = pickle.load(f, encoding = 'bytes')
    queries = Query_File
    start = time.time()
    candidates = query(queries, codebooks, codes, T=10)