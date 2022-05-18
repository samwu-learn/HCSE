import math
import heapq
#import numba as nb
import numpy as np
import copy
from collections import deque
import HCSE
def cut_volume(adj_matrix,S,T):
    if len(adj_matrix) == 0 or len(S) == 0 or len(T) == 0:
        return 0
    cut_vol = 0
    for u in S:
        for v in T:
            if adj_matrix[u][v] != 0:
               cut_vol += adj_matrix[u][v]
    return cut_vol

def merge(tree_node,new_ID, id1, id2, cut_v, node_list):
    new_vertices = tree_node[id1].vertices + tree_node[id2].vertices
    vol = tree_node[id1].vol + tree_node[id2].vol
    g = tree_node[id1].g + tree_node[id2].g - 2 * cut_v
    child_h = max(tree_node[id1].child_h, tree_node[id2].child_h) + 1
    parent = tree_node[id1].parent
    new_node = HCSE.PartitionTreeNode(ID=new_ID, vertices=new_vertices, children={id1, id2},
                                     parent=parent,g=g, vol=vol, child_h=child_h, child_cut=cut_v)
    tree_node[id1].parent, tree_node[id2].parent = new_ID, new_ID
    tree_node[new_ID]=new_node
    tree_node[parent].children.remove(id1)
    tree_node[parent].children.remove(id2)
    tree_node[parent].children.add(new_ID)
    node_list.append(new_ID)

def contain_vertices(A,B):
    d = [False for c in B if c not in A]
    if d:
        return False
    else:
        return True

def equal_vertices(A,B):
    if contain_vertices(A,B)==True and len(A)==len(B):
        return True
    else:
        return False

def exist_if(tree_node,vertices):
    for k,v in tree_node.items():
        if equal_vertices(v.vertices,vertices):
            return True
    return False

def stretch(tree_node,adj_matrix,adj_table,nodes,id_g):
    '''
    :param tree_node:
    :param adj_matrix:
    :param adj_table:
    :param nodes: 要stretch的u-triangle的所有孩子节点
    :param id_g:
    :return:
    '''
    ori_nodes = nodes
    nodes = list(nodes)
    root, min_heap = tree_node[nodes[0]].parent, []
    nodes.append(root)
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            if i == root or j == root:
                continue
            u, v = nodes[i], nodes[j]
            vol_u, vol_v = tree_node[u].vol, tree_node[v].vol
            cut = cut_volume(adj_matrix, tree_node[u].vertices, tree_node[v].vertices)
            reduce = -cut * math.log2(tree_node[root].vol / (vol_u + vol_v))
            if reduce == 0:
                continue
            heapq.heappush(min_heap, (reduce, u, v, cut))
    unmerged_number = len(nodes) - 1
    for node in nodes:
        if node != root:
            tree_node[node].merged = False
    while unmerged_number > 1:
        if len(min_heap) == 0:
            break
        reduce, u, v, cut = heapq.heappop(min_heap)
        if tree_node[u].merged or tree_node[v].merged:
            continue
        if reduce >= 0:
            break
        if exist_if(tree_node, tree_node[u].vertices + tree_node[v].vertices) == True:
            continue
        tree_node[u].merged, tree_node[v].merged = True, True
        unmerged_number -= 1
        if unmerged_number % 500 == 0:
            print('unmerged number:', unmerged_number)
            print('min_heap:', len(min_heap))
        new_id = next(id_g)
        merge(tree_node, new_id, u, v, cut, nodes)
        adj_table[new_id] = adj_table[u].union(adj_table[v])
        root, u = tree_node[new_id].parent, tree_node[new_id]
        for i in range(len(nodes)):
            v = nodes[i]
            if tree_node[v].merged == True:
                continue
            if v in u.children or v == new_id:  ##change： v in u.vertices 改成 v in u.children
                continue
            vol_u, vol_v = u.vol, tree_node[v].vol
            cut = cut_volume(adj_matrix, tree_node[v].vertices, u.vertices)
            reduce = -cut * math.log2(tree_node[root].vol / (vol_u + vol_v))
            if reduce == 0:
                continue
            heapq.heappush(min_heap, (reduce, new_id, v, cut))
        nodes.append(new_id)
    nodes.append(root)
    return root, list(set(nodes))  # 把此u-triangle的root和nodes输出保存起来，以在compress里使用
