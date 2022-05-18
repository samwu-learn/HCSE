import math
import heapq
#import numba as nb
import numpy as np
import copy
from collections import deque

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

def compressdelta(tree_node,parent,now,children,nodes):
    '''
    :param parent:  now的parent
    :param now: 当前要删除的节点
    :param children: now的孩子节点
    :param nodes: u-triangle上节点
    :return: now节点被删除后，entropy的增长值，正比于(g_child-g)*math.log2(tree_node[parent].vol/tree_node[now].vol)
    '''
    g_child = 0.0
    if not children:
        return 0.0
    for child in children:
        if child in nodes:
            g_child += tree_node[child].g
    g = g_child-tree_node[now].g
    return g*math.log2(tree_node[parent].vol/tree_node[now].vol)

def compress(tree_node,root,ori_nodes,vol_all):
    '''
    :param tree_node: 整棵树的树节点字典
    :param root: stretch后T_u的root id
    :param id_g: 当前id
    :param ori_nodes: T_u的原始节点，用以找树高
    :return: compress此子树后的树root
    '''
    nodes = set(ori_nodes)#转成set减少查找时间
    min_heap = []
    for i in nodes:
        if i==root:
            continue
        v = tree_node[i]
        parent,children= v.parent,v.children#考虑计算删除v对 entropy的影响
        if not v.children:
            continue
        delta = compressdelta(tree_node,parent,i,children,nodes)
        heapq.heappush(min_heap,(delta,i))
    while True:
        if get_height_of_subtree(tree_node,root,nodes)<=2:
            break
        if not min_heap:
            break
        delta,i = heapq.heappop(min_heap)
        v = tree_node[i]
        parent,children = v.parent,v.children
        if not children:
            continue
        u = tree_node[parent]
        if i in u.children:
            u.children.remove(i)
        for child in children:
            u.children.add(child)
            tree_node[child].parent = parent
        del tree_node[i]
        nodes.remove(i)
        #update delta of children(children的父节点发生了改变，delta值需要调整)
        # update delta of parent(parent的孩子节点发生改变)
        for i in range(len(min_heap)):
            delta,index = min_heap[i]
            if index in children:
                del min_heap[i]
                delta = compressdelta(tree_node, tree_node[index].parent, index, tree_node[index].children, nodes)
                heapq.heappush(min_heap, (delta, index))
            if index==parent:
                del min_heap[i]
                delta = compressdelta(tree_node, tree_node[index].parent, index, tree_node[index].children, nodes)
                heapq.heappush(min_heap, (delta, index))
    return nodes
