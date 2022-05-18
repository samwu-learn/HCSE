import math
import heapq
#import numba as nb
import numpy as np
import copy
from collections import deque
import stretch
import compress

def get_height_of_subtree(tree_node,root,nodes):
    '''
    :param root: 当前子树的根节点
    :param nodes: stretch树的所有子节点构成的集合，用以规定范围
    :return: 树高 height
    '''
    if not root:
        return 0
    height,queue = -1,deque()
    queue.append(root)
    while queue:
        n = len(queue)
        for _ in range(n):
            p = queue.popleft()
            if tree_node[p].children:
                for child in tree_node[p].children:
                    if child in nodes:
                        queue.append(child)
                    else:
                        queue.append(child)
        height += 1
    return height

def get_entropy(tree_node,root,nodes,vol_all):
    '''
    :param tree_node: 树节点字典
    :param root:  当前u-triangle的根节点
    :param nodes: 当前u-triangle的结点
    :return:      树结构熵
    '''
    entropy = 0.0
    for node in nodes:
        if node == root:
            continue
        u = tree_node[node]
        parent = tree_node[u.parent]
        if u.vol!=0:
            entropy += (u.g/vol_all)*math.log2(parent.vol/u.vol)
    return entropy

def check_level(tree_node,layer_node):
    for i in layer_node:
        if tree_node[i].children != None and len(tree_node[i].children)!= 2:
            return False
    return True

def find_sparset_level(tree_node,adj_matrix,adj_table,root,id_g,vol_all,last_layer):
    '''
    :param tree_node:  整棵树的所有节点
    :param adj_matrix: 图的邻接矩阵
    :param adj_table:  每个点的邻接情况
    :param id_g:       当前的id数
    :param vol_all:    根节点的容量
    :param last_layer: 上次stretch、compress的层，本次不能再此层
    :return:           找到sparest的一层，且保存其他所有层的spar值
    '''
    layer_nodes,layer,id_max= {},0,0#按层(0,1,2,...,k)存储节点信息
    my_queue = deque()
    my_queue.append(root)
    while my_queue:
        n = len(my_queue)
        now_layer = set()
        for _ in range(n):
            p = my_queue.popleft()
            now_layer.add(p)
            if tree_node[p].children:
                for child in tree_node[p].children:
                    my_queue.append(child)
            if tree_node[p].ID>id_max:
                id_max = tree_node[p].ID
        layer_nodes[layer] = now_layer
        layer += 1
    #保存原树的信息，再通过遍历的方式找sparest level
    sparse, reduce_hu= {},{}
    ori_height = get_height_of_subtree(tree_node,root,list(tree_node.keys()))
    for i in range(0,layer-1):
        sparse_i, reduce_hu_i = 0.0, [0.0]
        now = set(layer_nodes[i])
        if check_level(tree_node,now):
            sparse[i] = sparse_i
            continue
        ori_tree_node, ori_adj_table = copy.deepcopy(tree_node), copy.deepcopy(adj_table)
        for j in now:
            u,ori_Hu = ori_tree_node[j],0.0
            if u.children:
                for child in u.children:
                    if ori_tree_node[child].vol!=0:
                        ori_Hu += (ori_tree_node[child].g/vol_all) * math.log2(u.vol/ori_tree_node[child].vol)
                now_root,nodes = stretch.stretch(ori_tree_node,adj_matrix,ori_adj_table,u.children,id_g)
                nodes = compress.compress(ori_tree_node,now_root,nodes,vol_all)
                after_se = get_entropy(ori_tree_node,now_root,nodes,vol_all)
                spar_u = (ori_Hu-after_se)/ori_Hu
                reduce_hu_i += [ori_Hu-after_se]
                sparse_i += spar_u
        height = get_height_of_subtree(ori_tree_node,root,list(ori_tree_node.keys()))
        if height-ori_height<=0:
            sparse[i] = 0.0
        else:
            sparse[i] = sparse_i/len(now)
        reduce_hu[i] = reduce_hu_i
    max_layer,max_delta= -1,0.0
    for lay in sparse.keys():
        if sparse[lay]>max_delta:
            max_delta = sparse[lay]
            max_layer = lay
    sparsest = max_delta
    return max_layer,layer_nodes,sparsest