import scipy.spatial
import numpy as np
import copy
import heapq
def pq(data, P, init_centroids, max_iter):
    #split_data,,将input数据的维度平均切割成p份
    dimensionality = len(data[0])
    dimensionality_after_split = int(dimensionality/P)
    data_after_split = []
    flag = 0
    data = np.array(data)
    while P > 0:
        split_data_list = data[:,flag:flag + dimensionality_after_split]
        flag += dimensionality_after_split
        P -= 1
        data_after_split.append(split_data_list)
    codebooks = []
    k = len(init_centroids)#k是中心点的数量，即表示有多少个分类
    #开始对每一个分割后的数据集进行处理
    for i in range(len(data_after_split)):
        #复制初始的中心点
        tem_center = copy.deepcopy(init_centroids[i])
        #kmeans的循环最大次数为max_iter
        tem_max_iter = max_iter
        while tem_max_iter > 0:
            tem_max_iter -= 1
            #用来存储每个中心点包含的vectors，由于中心点在变化，所以每次都需要清空
            center_cluster = []
            for num in range(len(tem_center)):
                center_cluster.append([])
            #获取每个点与所有center之间的L1距离
            distance_list = scipy.spatial.distance.cdist(data_after_split[i], tem_center, metric='cityblock')
            #获取每个vector对应的center的index，该index对应的是center_list中的index
            min_distance_index = np.argmin(distance_list,axis=1)
            #将vector分配到其所属的center中
            for num in range(len(data_after_split[i])):
                center_point_index = min_distance_index[num]#获取该点对应的中心点index
                vector = data_after_split[i][num]#获取该点
                center_cluster[center_point_index].append(vector)#将该点加入中心点对应的cluster
            #重新计算中心点
            old_center = copy.deepcopy(tem_center)#复制中心点用于比较
            for num in range(len(center_cluster)):
                #如果该中心点cluster有对应的点加入，重新计算中心点
                if len(center_cluster[num]) > 0:
                    tem_center[num] = np.median(center_cluster[num],axis=0)
                else:
                    #没有对应的点加入，中心点保持不变
                    continue
            #与上一次循环的中心点进行比较，若完全相同，可提前跳出循环
            if((old_center == tem_center).all()):
                break
            else:
                continue
        #每一轮kmeans结束后，将得到的中心点及对应的code将入结果集
        codebooks.append(tem_center)
    #生成codes
    codes = []
    for i in range(len(data_after_split)):
        #计算与每个点的距离
        distance_list = scipy.spatial.distance.cdist(data_after_split[i], codebooks[i], metric='cityblock')
        min_distance_index = np.argmin(distance_list,axis=1)
        #初始化codes
        tem_codes = np.linspace(-1,-1,len(data_after_split[i]))
        for num in range(len(data_after_split[i])):
            center_point_index = min_distance_index[num]
            tem_codes[num] = center_point_index#设置该点对应中心点的index
        codes.append(tem_codes)
    codebooks = np.array(codebooks,dtype = 'float32')
    codes = np.array(codes,dtype = 'uint8')
    #将矩阵进行旋转
    codes = np.transpose(codes)
    return codebooks, codes
    
def query(queries, codebooks, codes, T):
    #切割query
    for i in range(len(codes)):
        codes[i] = tuple(codes[i])
    P = len(codebooks)
    dimensionality = len(queries[0])
    dimensionality_after_split = int(dimensionality/P)
    query_after_split = []
    queries = np.array(queries)
    for one_query in queries:
        tem_p = P
        flag = 0
        split_query_list = []
        while tem_p > 0:
            split_query_list.append(one_query[flag:flag + dimensionality_after_split])
            flag += dimensionality_after_split
            tem_p -= 1
        query_after_split.append(split_query_list)
    #query切割完成
    #计算query点与codebooks点之间L1距离
    distance_set = []
    for one_query in query_after_split:
        one_query_distance_set = []
        for i in range(len(one_query)):
            one_query_distance = scipy.spatial.distance.cdist([one_query[i]], codebooks[i], metric='cityblock')
            one_query_distance_set.append(one_query_distance)
        distance_set.append(one_query_distance_set)
    distance_set = np.array(distance_set)
    #print(distance_set.shape)
    #完成距离计算
    #存储codebook中的index对应的distance，并按从小到大的顺序排序
    #index_and_distance = []
    #生成的index是distance_set中从小到大排序的index
    all_distance_sort_index = []
    for sub_distance_set in distance_set:
        tem_distance_sort_index = []
        for one_distance in sub_distance_set:
            ##print(np.argsort(one_distance))
            tem_distance_sort_index.append(np.argsort(one_distance))
        all_distance_sort_index.append(tem_distance_sort_index)
    #all_distance_sort_index = np.array(all_distance_sort_index)
    ##print(all_distance_sort_index.shape)
    ##print(index_and_distance[0][0])
    #生成location_index,字典的key为坐标，value为codes中的index
    location_index = {}
    for i in range(codes.shape[0]):
        location = tuple(codes[i])
        if location not in location_index:
            location_index[location] = {i}
        else:
            location_index[location].add(i)
    #print(location_index)
    all_output = []
    for num in range(len(all_distance_sort_index)):
        traversed_dic = {}
        tem_index_of_index_and_distance = []
        tem_dist = 0
        pqueue = []
        for i in range(len(all_distance_sort_index[num])):
            tem_index_of_index_and_distance.append(0)
            tem_location = all_distance_sort_index[num][i][0][0]
            tem_dist += distance_set[num][i][0][tem_location]
            ##print(i)
            ##print(distance_set[num][i][0][tem_location])
        tem_index_of_index_and_distance = tuple(tem_index_of_index_and_distance)
        output = []
        heapq.heappush(pqueue,(tem_dist,tem_index_of_index_and_distance))
        traversed_dic[tem_index_of_index_and_distance] = 1
        while len(output)<T:
            pop_result = heapq.heappop(pqueue)
            tem_index = pop_result[1]
            #print(f'pop:{tem_index}')
            index_in_codes = []
            for i in range(len(tem_index)):
                #print(all_distance_sort_index[num][i][0][tem_index[i]])
                index_in_codes.append(all_distance_sort_index[num][i][0][tem_index[i]])
            #print(index_in_codes)
            index_in_codes = tuple(index_in_codes)
            if index_in_codes in location_index:
                output.extend(location_index[index_in_codes])
            P = len(codebooks)
            for p_num in range(P):
                if tem_index[p_num] < len(all_distance_sort_index[num][i][0])-1:
                    traversed_judge_index = copy.deepcopy(tem_index)
                    traversed_judge_index = list(traversed_judge_index)
                    traversed_judge_index[p_num] += 1
                    if tuple(traversed_judge_index) not in traversed_dic:
                        tem_sum = 0
                        for b in range(len(traversed_judge_index)):
                            tem_location = all_distance_sort_index[num][b][0][traversed_judge_index[b]]
                            tem_sum += distance_set[num][b][0][tem_location]
                        heapq.heappush(pqueue,(tem_sum,tuple(traversed_judge_index)))
                        traversed_dic[tuple(traversed_judge_index)] = 1
        all_output.append(set(output))
    return all_output