import math
import heapq
#import numba as nb
import numpy as np
import copy
from collections import deque
import stretch
import compress
import find_k_sparest

def get_id():
    i = 0
    while True:
        yield i
        i += 1
def graph_parse(adj_matrix):
    num_of_nodes = adj_matrix.shape[0]
    adj_table = {}
    vol,node_vol = 0,[]
    for v in range(num_of_nodes):
        d_v = 0
        adj = set()
        for j in range(num_of_nodes):
            if adj_matrix[v][j] != 0:
                d_v += adj_matrix[v][j]
                vol += adj_matrix[v][j]
                adj.add(j)
        adj_table[v] = adj
        node_vol.append(d_v)
    return num_of_nodes,vol,node_vol,adj_table

#@nb.jit(nopython=True)
def cut_volume(adj_matrix,S,T):
    if len(adj_matrix) == 0 or len(S) == 0 or len(T) == 0:
        return 0
    cut_vol = 0
    for u in S:
        for v in T:
            if adj_matrix[u][v] != 0:
               cut_vol += adj_matrix[u][v]
    return cut_vol

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

def cost_se(tree_node,adj_matrix,root):
    se = 0.0
    for i in range(len(adj_matrix)-1):
        for j in range(i+1,len(adj_matrix)):
            if adj_matrix[i][j]!=0:
                parent = find_lca(tree_node,i,j,root)
                se += adj_matrix[i][j]*math.log2(tree_node[parent].vol)
    return se

def cost_das(tree_node,adj_matrix,root):
    das = 0
    for i in range(len(adj_matrix)-1):
        for j in range(i+1,len(adj_matrix)):
            if adj_matrix[i][j]!=0:
                parent = find_lca(tree_node,i,j,root)
                das += adj_matrix[i][j]*len(tree_node[parent].vertices)
    return das
class PartitionTreeNode():
    def __init__(self, ID, vertices, vol, g, children:set = None,parent = None,child_h = 0, child_cut = 0):
        self.ID = ID
        self.vertices = vertices
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h #不包括该节点的子树高度
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree():
    def __init__(self,adj_matrix,log_file=None):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()
        self.log_file=log_file
    def build_leaves(self):
        vertices,children = [],set()
        vol_all = 0
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            vol = self.node_vol[vertex]
            vol_all += vol
            leaf_node = PartitionTreeNode(ID=ID, vertices=[vertex], g = vol, vol=vol)
            vertices.append(vertex)
            children.add(ID)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)
        root_id = next(self.id_g)
        root = PartitionTreeNode(ID=root_id,vertices=vertices,vol=vol_all,g=vol_all,children=children,child_h=1)
        self.tree_node[root_id] = root
        for leaf in self.leaves:
            self.tree_node[leaf].parent = root_id
        self.root_id = root_id

    def build_tree(self):
        saved = {}
        delta = {}
        H = {}
        saved_entropy = get_entropy(self.tree_node, self.root_id, list(self.tree_node.keys()), self.VOL)
        saved_se = cost_se(self.tree_node, self.adj_matrix, self.root_id)
        saved_das = cost_das(self.tree_node, self.adj_matrix, self.root_id)
        saved_tree = copy.deepcopy(self.tree_node)
        saved[0] = [saved_entropy,saved_se,saved_das,saved_tree]
        saved_id = -1
        h=1
        now_root, nodes = stretch.stretch(self.tree_node, self.adj_matrix,self.adj_table, self.leaves, self.id_g)
        compress.compress(self.tree_node, now_root, nodes, self.VOL)

        saved_entropy = get_entropy(self.tree_node, self.root_id, list(self.tree_node.keys()), self.VOL)
        saved_se = cost_se(self.tree_node, self.adj_matrix, self.root_id)
        saved_das = cost_das(self.tree_node, self.adj_matrix, self.root_id)
        saved_tree = copy.deepcopy(self.tree_node)
        saved[1] = [saved_entropy, saved_se, saved_das, saved_tree]

        delta[0] = saved[1][0]-saved[0][0]
        last_layer, l, h = -1, 2, 2
        while True:
            max_layer,layer_nodes,sparsest = find_k_sparest.find_sparset_level(self.tree_node,self.adj_matrix,self.adj_table,self.root_id,self.id_g,self.VOL,last_layer)
            print('sparse',sparsest,'for ',max_layer)
            if sparsest<=0.0:
                saved_id = l-1
                break
            for i in layer_nodes[max_layer]:
                u = self.tree_node[i]
                if u.children:
                    now_root, nodes = stretch.stretch(self.tree_node, self.adj_matrix, self.adj_table, u.children, self.id_g)
                    compress.compress(self.tree_node, now_root, nodes, self.VOL)
            saved_entropy = get_entropy(self.tree_node, self.root_id, list(self.tree_node.keys()), self.VOL)
            saved_se = cost_se(self.tree_node, self.adj_matrix, self.root_id)
            saved_das = cost_das(self.tree_node, self.adj_matrix, self.root_id)
            saved_tree = copy.deepcopy(self.tree_node)
            saved[l] = [saved_entropy, saved_se, saved_das, saved_tree]
            delta[l-1] = saved[l][0]-saved[l-1][0]
            H[l-2] = delta[l-1]-delta[l-2]
            print('l is :',l)
            if l>3:
                if H[l-3]>H[l-2] and H[l-3]>H[l-4]:
                    saved_id = l-1
                    break
            last_layer = max_layer
            nodes = list(self.tree_node.keys())
            h = find_k_sparest.get_height_of_subtree(self.tree_node,self.root_id,nodes)
            l+=1
        print('saved_id:',saved_id)
        print('entropy:',saved[saved_id][0])
        print('se:',saved[saved_id][1])
        print('das:',saved[saved_id][2])
        self.tree_node = saved[saved_id][3]

def find_lca(tree_node,i,j,root_id):
    p= root_id
    flag = True
    while flag:
        flag = False
        for child in tree_node[p].children:
            if (i in tree_node[child].vertices)==True and (j in tree_node[child].vertices)==True:
                p = child
                flag=True
                continue
        if flag==False:
            break
    return p
