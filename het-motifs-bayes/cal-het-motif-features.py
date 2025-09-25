import math

import networkx as nx
import pandas as pd

#构建网络
f = open('MGTAB_edge_index_followers_friendes.txt')
net = []
for i in f:
    t = i.strip('\n').split(',')
    net.append([t[0],t[1]])
G = nx.DiGraph()
G.add_edges_from(net)
node_list = list(G.nodes)

edge_list = list(G.edges)
print("node num ",len(node_list),"    edge num ",len(edge_list))


#赋予标签
f_label = open('MGTAB_label.txt')
users_label = {}
for i in f_label:
    t = i.strip('\n').split(',')
    users_label[t[0]]=int(t[1])
print('len users_label:',len(users_label))

c_rob = 0 #计算网络中机器人节点个数
c_hum = 0
node_hum = []
node_rob = []
for i in users_label.keys():
    if (users_label[i] == 1) and (i in node_list):
        node_rob.append(i)
        c_rob += 1
    elif (users_label[i] == 0) and (i in node_list):
        node_hum.append(i)
        c_hum += 1
print('网络中机器人节点个数：',c_rob)
print('网络中人类节点个数：',c_hum)

a = 6968/2475   #网络中人类节点与机器人节点数量之比
def has_DiEdge(G,node1,node2):
    nbr1=list(G.predecessors(node2))
    if node1  in nbr1:
        return True
    else:
        return False




def find_M1(G, A, B, node):
    nodes_pointing_to_A = list(G.predecessors(A))

    # 获取所有指向B的节点
    nodes_pointing_to_B = list(G.predecessors(B))

    # 找出同时指向A和B的节点
    common_nodes = set(nodes_pointing_to_A).intersection(set(nodes_pointing_to_B))

    # 过滤掉A指向的节点和B指向的节点
    result = [node for node in common_nodes if not G.has_edge(A, node) and not G.has_edge(B, node)]
    if node in result:
        result.remove(node)

    return result

def caculate_M1_log(node,num):
    c_log = 0
    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M1(G, parir_i[1], parir_i[2], node)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M1(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.successors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (not has_DiEdge(G, nbr1[i], nbr1[j])) and (not has_DiEdge(G, nbr1[j], nbr1[i])) and (
            not has_DiEdge(G, nbr1[j], node)) and (not has_DiEdge(G, nbr1[i], node)):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0 :
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1 :
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or  (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0) :
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M1_log(node,num1),caculate_M1_log(node,num2),caculate_M1_log(node,num3)



def find_M2(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if C in successors_of_B and not G.has_edge(A, C) and not G.has_edge(C, B)]
    if node in result:
        result.remove(node)

    return result

def caculate_M2_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M2(G, parir_i[1], parir_i[2], node)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M2(G, node):
    num1 = [];num2 = [];num3 = [];num4 = [];
    nbr1 = list(G.predecessors(node))
    nbr2 = list(G.successors(node))
    for i in nbr1:
        for j in nbr2:
            if (i not in list(G.predecessors(j))) and (i not in list(G.successors(j))) and (
                    not has_DiEdge(G, node, i) and (not has_DiEdge(G, j, node))):
                if users_label[i] == 0 and users_label[j] == 0:
                    num1.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 1:
                    num2.append([node, i, j])
                if (users_label[i] == 0 and users_label[j] == 1) :
                    num3.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 0:
                    num4.append([node, i, j])
    return caculate_M2_log(node, num1), caculate_M2_log(node, num2), caculate_M2_log(node, num3),caculate_M2_log(node, num4)


def find_M3(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in successors_of_B and not G.has_edge(A, C) and G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M3_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        node_list_1 = find_M3(G, parir_i[1], parir_i[2], node)
        node_list_2 = find_M3(G, parir_i[2], parir_i[1], node)
        all_node_list = list(set(node_list_1 + node_list_2))
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M3(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    nbr2 = list(G.predecessors(node))
    for i in nbr1:
        for j in nbr2:
            if (i not in list(G.predecessors(j))) and (i not in list(G.successors(j))) and (node in G.predecessors(j)):
                if (not has_DiEdge(G, i, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M3_log(node, num1), caculate_M3_log(node, num2), caculate_M3_log(node, num3), caculate_M3_log(node,
                                                                                                                  num4)
def find_M4(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在网络G中")

    predecessors_of_A = list(G.predecessors(A))
    successors_of_B = list(G.successors(B))

    result = []
    for C in predecessors_of_A:
        if C in successors_of_B and not G.has_edge(A, C) and not G.has_edge(C, B):
            result.append(C)
    if node in result:
        result.remove(node)
    return result

def caculate_M4_log(node,num):
    c_log = 0
    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M4(G, parir_i[1], parir_i[2], node)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M4(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1)-1):
        for j in range(i+1,len(nbr1)):
            if (not has_DiEdge(G,nbr1[i],nbr1[j])) and (not has_DiEdge(G,nbr1[j],nbr1[i])) and (not has_DiEdge(G,node,nbr1[j])) and (not has_DiEdge(G,node,nbr1[i])):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M4_log(node, num1), caculate_M4_log(node, num2), caculate_M4_log(node,num3)

def find_M5(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and not G.has_edge(C, A) and G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M5_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        node_list_1 = find_M5(G, parir_i[1], parir_i[2], node)
        node_list_2 = find_M5(G, parir_i[2], parir_i[1], node)
        all_node_list = list(set(node_list_1 + node_list_2))
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M5(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (not has_DiEdge(G, nbr1[i], nbr1[j])) and (not has_DiEdge(G, nbr1[j], nbr1[i])):
                if (not has_DiEdge(G, node, nbr1[j])) and has_DiEdge(G, node, nbr1[i]):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num4.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0:
                        num3.append([node, nbr1[i], nbr1[j]])
                elif has_DiEdge(G, node, nbr1[j]) and (not has_DiEdge(G, node, nbr1[i])):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num3.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0:
                        num4.append([node, nbr1[i], nbr1[j]])
    return caculate_M5_log(node, num1), caculate_M5_log(node, num2), caculate_M5_log(node, num3), caculate_M5_log(node, num4)

def find_M6(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and  G.has_edge(C, A) and G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M6_log(node,num):
    c_log = 0
    for parir_i in num:
        count_human = 0
        count_rob = 0
        node_list_1 = find_M6(G, parir_i[1], parir_i[2], node)
        node_list_2 = find_M6(G, parir_i[2], parir_i[1], node)
        all_node_list = list(set(node_list_1 + node_list_2))
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M6(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (not has_DiEdge(G, nbr1[i], nbr1[j])) and (not has_DiEdge(G, nbr1[j], nbr1[i])) and (
            has_DiEdge(G, node, nbr1[j])) and (has_DiEdge(G, node, nbr1[i])):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M6_log(node, num1), caculate_M6_log(node, num2), caculate_M6_log(node, num3)

def find_M7(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in successors_of_B and  G.has_edge(A, C) and G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M7_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M7(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M7(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if  not(has_DiEdge(G,i,node)):
            nbr2=list(G.predecessors(i))
            for j in nbr2:
                if  (not has_DiEdge(G,i,j)) and  has_DiEdge(G,j,node) and (not has_DiEdge(G,node,j)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M7_log(node, num1), caculate_M7_log(node, num2), caculate_M7_log(node,num3), caculate_M7_log(node,num4)

def find_M8(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in predecessors_of_B and  not G.has_edge(A, C) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M8_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M8(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M8(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (not has_DiEdge(G, nbr1[i], node)) and (not has_DiEdge(G, nbr1[j], node)):
                    if has_DiEdge(G, nbr1[i],nbr1[j]) and (not has_DiEdge(G, nbr1[j], nbr1[i])):
                        if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                            num1.append([node, nbr1[i], nbr1[j]])
                        if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                            num2.append([node, nbr1[i], nbr1[j]])
                        if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) :
                            num3.append([node, nbr1[i], nbr1[j]])
                        if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                            num4.append([node, nbr1[i], nbr1[j]])
                    elif ((not has_DiEdge(G,nbr1[i],nbr1[j])) and has_DiEdge(G, nbr1[j], nbr1[i])):
                        if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                            num1.append([node, nbr1[i], nbr1[j]])
                        if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                            num2.append([node, nbr1[i], nbr1[j]])
                        if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                            num4.append([node, nbr1[i], nbr1[j]])
                        if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                            num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M8_log(node, num1), caculate_M8_log(node, num2), caculate_M8_log(node, num3), caculate_M8_log(node,
                                                                                                                  num4)
def find_M9(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and   G.has_edge(C, A) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M9_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M9(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)
def M9(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if ((not has_DiEdge(G, nbr1[i], node)) and has_DiEdge(G, nbr1[j], node) and has_DiEdge(G, nbr1[j], nbr1[i])
                    and (not has_DiEdge(G, nbr1[i], nbr1[j]))):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[j], nbr1[i]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[j], nbr1[i]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                    num4.append([node, nbr1[j], nbr1[i]])
                if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[j], nbr1[i]])
            elif (has_DiEdge(G, nbr1[i], node) and not has_DiEdge(G, nbr1[j], node) and has_DiEdge(G, nbr1[i], nbr1[j])
                  and not has_DiEdge(G, nbr1[j], nbr1[i])):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                    num3.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num4.append([node, nbr1[i], nbr1[j]])
    return caculate_M9_log(node, num1), caculate_M9_log(node, num2), caculate_M9_log(node, num3), caculate_M9_log(node,num4)

def find_M10(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in predecessors_of_B and not G.has_edge(C, A) and not G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M10_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M10(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M10(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if has_DiEdge(G, nbr1[i], nbr1[j]) and has_DiEdge(G, nbr1[j], nbr1[i]) and (
                    not has_DiEdge(G, node, nbr1[j])) and (not has_DiEdge(G, node, nbr1[i])):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or (
                        users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M10_log(node, num1), caculate_M10_log(node, num2), caculate_M10_log(node, num3)

def find_M11(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in successors_of_B and not G.has_edge(C, A) and not G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M11_log(node,num):
    c_log = 0
    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M11(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M11(G, node):
    num1 = [];num2 = [];num3 = [];num4 = [];
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (not has_DiEdge(G, node, nbr1[j])) and (not has_DiEdge(G, node, nbr1[i])):
                if (has_DiEdge(G, nbr1[i],nbr1[j]) and not has_DiEdge(G, nbr1[j], nbr1[i])):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num3.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                        num4.append([node, nbr1[i], nbr1[j]])
                elif not has_DiEdge(G, nbr1[i],nbr1[j]) and has_DiEdge(G, nbr1[j], nbr1[i]):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[j], nbr1[i]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[j], nbr1[i]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num4.append([node, nbr1[j], nbr1[i]])
                    if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                        num3.append([node, nbr1[j], nbr1[i]])
    return caculate_M11_log(node, num1), caculate_M11_log(node, num2), caculate_M11_log(node, num3), caculate_M11_log(node, num4)

def find_M12(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in successors_of_B and not G.has_edge(A, C) and not G.has_edge(C,B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M12_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M12(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M12(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (not has_DiEdge(G, i, node)):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, j, i)) and (not has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M12_log(node, num1), caculate_M12_log(node, num2), caculate_M12_log(node, num3), caculate_M12_log(node, num4)

def find_M13(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and not G.has_edge(C, A) and not G.has_edge(B,C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M13_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        node_list_1 = find_M13(G, parir_i[1], parir_i[2], node)
        node_list_2 = find_M13(G, parir_i[2], parir_i[1], node)
        all_node_list = list(set(node_list_1 + node_list_2))
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M13(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (not has_DiEdge(G, i, node)):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (has_DiEdge(G, j, i)) and (not has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M13_log(node, num1), caculate_M13_log(node, num2), caculate_M13_log(node, num3), caculate_M13_log(node, num4)

def find_M14(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in successors_of_B and G.has_edge(C, A) and not G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M14_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M14(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M14(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (has_DiEdge(G, i, node)):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, j, i)) and (not has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M14_log(node, num1), caculate_M14_log(node, num2), caculate_M14_log(node, num3), caculate_M14_log(node, num4)


def find_M15(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in successors_of_B and not G.has_edge(A, C) and  G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M15_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M15(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M15(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (not has_DiEdge(G, i, node)):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, j, i)) and ( has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M15_log(node, num1), caculate_M15_log(node, num2), caculate_M15_log(node, num3), caculate_M15_log(node, num4)

def find_M16(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if
              C in predecessors_of_B and not G.has_edge(A, C) and  not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M16_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M16(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M16(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.successors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (has_DiEdge(G, nbr1[i], nbr1[j])) and (has_DiEdge(G, nbr1[j], nbr1[i])) and (
            not has_DiEdge(G, nbr1[j], node)) and (not has_DiEdge(G, nbr1[i], node)):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or (
                        users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M16_log(node, num1), caculate_M16_log(node, num2), caculate_M16_log(node, num3)

def find_M17(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and not G.has_edge(C, A) and   G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M17_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M17(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M17(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in nbr1:
        if (not has_DiEdge(G, node, i)):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, j, i)) and (has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M17_log(node, num1), caculate_M17_log(node, num2), caculate_M17_log(node, num3), caculate_M17_log(node, num4)

def find_M18(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    predecessors_of_B = set(G.predecessors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in predecessors_of_B and  G.has_edge(C, A) and  not  G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M18_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M18(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M18(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in nbr1:
        if (has_DiEdge(G, node, i)):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (has_DiEdge(G, i, j)) and (has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M18_log(node, num1), caculate_M18_log(node, num2), caculate_M18_log(node, num3), caculate_M18_log(node, num4)

def find_M19(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in successors_of_B and  G.has_edge(C, A) and  not  G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M19_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M19(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)


def M19(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (has_DiEdge(G, i, node)):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (has_DiEdge(G, i, j)) and (not has_DiEdge(G, node, j)) and (has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M19_log(node, num1), caculate_M19_log(node, num2), caculate_M19_log(node, num3), caculate_M19_log(node, num4)

def find_M20(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in successors_of_B and  G.has_edge(C, A) and   G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M20_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M20(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M20(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if   has_DiEdge(G, node,nbr1[j]) and has_DiEdge(G, node, nbr1[i]):
                if has_DiEdge(G,nbr1[i],nbr1[j]) and not has_DiEdge(G,nbr1[j],nbr1[i]):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[i], nbr1[j]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num3.append([node, nbr1[i], nbr1[j]])
                    if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                        num4.append([node, nbr1[i], nbr1[j]])
                elif has_DiEdge(G,nbr1[j],nbr1[i]) and not has_DiEdge(G,nbr1[i],nbr1[j]):
                    if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                        num1.append([node, nbr1[j], nbr1[i]])
                    if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                        num2.append([node, nbr1[j], nbr1[i]])
                    if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1):
                        num4.append([node, nbr1[j], nbr1[i]])
                    if (users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                        num3.append([node, nbr1[j], nbr1[i]])
    return caculate_M20_log(node, num1), caculate_M20_log(node, num2), caculate_M20_log(node, num3), caculate_M20_log(node, num4)

def find_M21(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 获取从B出发的节点（B的后继节点）
    successors_of_B = set(G.successors(B))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if
              C in successors_of_B and  G.has_edge(C, A) and   G.has_edge(C, B)]
    if node in result:
        result.remove(node)
    return result

def caculate_M21_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M21(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M21(G, node):
    num1 = [];num2 = [];num3 = []
    nbr1 = list(G.predecessors(node))
    for i in range(len(nbr1) - 1):
        for j in range(i + 1, len(nbr1)):
            if (has_DiEdge(G, nbr1[i], nbr1[j])) and (has_DiEdge(G, nbr1[j], nbr1[i])) and (
            has_DiEdge(G, node, nbr1[j])) and (has_DiEdge(G, node, nbr1[i])):
                if users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 0:
                    num1.append([node, nbr1[i], nbr1[j]])
                if users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 1:
                    num2.append([node, nbr1[i], nbr1[j]])
                if (users_label[nbr1[i]] == 0 and users_label[nbr1[j]] == 1) or (
                        users_label[nbr1[i]] == 1 and users_label[nbr1[j]] == 0):
                    num3.append([node, nbr1[i], nbr1[j]])
    return caculate_M21_log(node, num1), caculate_M21_log(node, num2), caculate_M21_log(node, num3)

def find_M22(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if C!=B and
              not G.has_edge(C, A) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M22_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M22(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M22(G, node):
    num1 = [];num2 = [];num3 = [];num4 = [];
    nbr1 = list(G.predecessors(node))
    for i in nbr1:
        if (not has_DiEdge(G, node, i)):
            nbr2 = list(G.successors(i))
            if node in nbr2:
                del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, j, i)) and (not has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M22_log(node, num1), caculate_M22_log(node, num2), caculate_M22_log(node, num3), caculate_M22_log(node, num4)

def find_M23(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if not G.has_edge(C, A) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M23_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M23(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M23(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in nbr1:
        nbr2 = list(G.predecessors(i))
        if node in nbr2:
            del nbr2[nbr2.index(node)]
        for j in nbr2:
            if (not has_DiEdge(G, node, i)) and (not has_DiEdge(G, i, j)) and (not has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                if users_label[i] == 0 and users_label[j] == 0:
                    num1.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 1:
                    num2.append([node, i, j])
                if (users_label[i] == 0 and users_label[j] == 1):
                    num3.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 0:
                    num4.append([node, i, j])
    return caculate_M23_log(node, num1), caculate_M23_log(node, num2), caculate_M23_log(node,num3), caculate_M23_log(node, num4)

def find_M24(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if not G.has_edge(A, C) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M24_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M24(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M24(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        nbr2 = list(G.successors(i))
        if node in nbr2:
            del nbr2[nbr2.index(node)]
        for j in nbr2:
            if (not has_DiEdge(G, i, node)) and (not has_DiEdge(G, j, i)) and (
                    not has_DiEdge(G, node, j) and (not has_DiEdge(G, j, node))):
                if users_label[i] == 0 and users_label[j] == 0:
                    num1.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 1:
                    num2.append([node, i, j])
                if (users_label[i] == 0 and users_label[j] == 1):
                    num3.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 0:
                    num4.append([node, i, j])
    return caculate_M24_log(node, num1), caculate_M24_log(node, num2), caculate_M24_log(node,num3), caculate_M24_log(node, num4)

def find_M25(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if C!=B and not G.has_edge(C, A) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M25_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M25(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M25(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.predecessors(node))
    for i in nbr1:
        nbr2 = list(G.successors(i))
        if node in nbr2: del nbr2[nbr2.index(node)]
        for j in nbr2:
            if (not has_DiEdge(G, node, i)) and has_DiEdge(G, j, i) and (not has_DiEdge(G, node, j)) and (
            not has_DiEdge(G, j, node)):
                if users_label[i] == 0 and users_label[j] == 0:
                    num1.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 1:
                    num2.append([node, i, j])
                if (users_label[i] == 0 and users_label[j] == 1):
                    num3.append([node, i, j])
                if users_label[i] == 1 and users_label[j] == 0:
                    num4.append([node, i, j])
    return caculate_M25_log(node, num1), caculate_M25_log(node, num2), caculate_M25_log(node,num3), caculate_M25_log(node, num4)

def find_M26(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if  G.has_edge(A, C) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M26_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M26(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M26(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if has_DiEdge(G, i, node):
            nbr2 = list(G.successors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if not (has_DiEdge(G, j, i)) and (not has_DiEdge(G, j, node)) and (not has_DiEdge(G, node, j)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M26_log(node, num1), caculate_M26_log(node, num2), caculate_M26_log(node,num3), caculate_M26_log(node, num4)

def find_M27(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    successors_of_A = set(G.successors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in successors_of_A if  G.has_edge(C, A) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M27_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M27(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M27(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if has_DiEdge(G, i, node):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, i, j)) and (not has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M27_log(node, num1), caculate_M27_log(node, num2), caculate_M27_log(node, num3), caculate_M27_log(node, num4)

def find_M28(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if B!=C and not G.has_edge(A, C) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M28_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M28(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M28(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if not (has_DiEdge(G, i, node)):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (not has_DiEdge(G, i, j)) and (not has_DiEdge(G, j, node)) and (not has_DiEdge(G, node, j)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M28_log(node, num1), caculate_M28_log(node, num2), caculate_M28_log(node,num3), caculate_M28_log(node, num4)

def find_M29(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if B!=C and not G.has_edge(A, C) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M29_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M29(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M29(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if (not has_DiEdge(G, i, node)):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if has_DiEdge(G, i, j) and (not has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M29_log(node, num1), caculate_M29_log(node, num2), caculate_M29_log(node,num3), caculate_M29_log(node, num4)

def find_M30(G, A, B, node):
    if A not in G or B not in G:
        raise ValueError(f"节点 {A} 或 {B} 不在图G中")

        # 获取指向A的节点（A的前驱节点）
    predecessors_of_A = set(G.predecessors(A))

    # 找出同时在前驱和后继集合中的节点，并检查额外的条件
    result = [C for C in predecessors_of_A if B!=C and  G.has_edge(A, C) and   not G.has_edge(C, B) and not G.has_edge(B, C)]
    if node in result:
        result.remove(node)
    return result

def caculate_M30_log(node,num):
    c_log = 0

    for parir_i in num:
        count_human = 0
        count_rob = 0
        all_node_list = find_M30(G, parir_i[1], parir_i[2], node)
        # print('all_node_list:', all_node_list)
        for list_i in all_node_list:
            if users_label[list_i] == 0:
                count_human += 1
            elif users_label[list_i] == 1:
                count_rob += 1
        c_log += math.log10((count_rob + 1) / (count_human + 1))
    return str(len(num) * math.log10(a) + c_log)

def M30(G, node):
    num1 = [];num2 = [];num3 = [];num4 = []
    nbr1 = list(G.successors(node))
    for i in nbr1:
        if has_DiEdge(G, i, node):
            nbr2 = list(G.predecessors(i))
            if node in nbr2: del nbr2[nbr2.index(node)]
            for j in nbr2:
                if (has_DiEdge(G, i, j)) and (not has_DiEdge(G, node, j)) and (not has_DiEdge(G, j, node)):
                    if users_label[i] == 0 and users_label[j] == 0:
                        num1.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 1:
                        num2.append([node, i, j])
                    if (users_label[i] == 0 and users_label[j] == 1):
                        num3.append([node, i, j])
                    if users_label[i] == 1 and users_label[j] == 0:
                        num4.append([node, i, j])
    return caculate_M30_log(node, num1), caculate_M30_log(node, num2), caculate_M30_log(node, num3), caculate_M30_log(node, num4)


motif_f = open('MGTAB_yi_motif.csv','w')

col = ['src', 'M1_1', 'M1_2', 'M1_3', 'M2_1', 'M2_2', 'M2_3', 'M2_4', 'M3_1', 'M3_2', 'M3_3', 'M3_4',
       'M4_1', 'M4_2', 'M4_3','M5_1', 'M5_2', 'M5_3', 'M5_4', 'M6_1', 'M6_2', 'M6_3', 'M7_1', 'M7_2', 'M7_3', 'M7_4',
       'M8_1', 'M8_2', 'M8_3', 'M8_4', 'M9_1', 'M9_2', 'M9_3', 'M9_4', 'M10_1', 'M10_2', 'M10_3',
       'M11_1', 'M11_2', 'M11_3', 'M11_4','M12_1', 'M12_2', 'M12_3', 'M12_4', 'M13_1', 'M13_2', 'M13_3', 'M13_4',
       'M14_1', 'M14_2', 'M14_3', 'M14_4', 'M15_1', 'M15_2', 'M15_3', 'M15_4', 'M16_1', 'M16_2', 'M16_3',
       'M17_1', 'M17_2', 'M17_3', 'M17_4', 'M18_1', 'M18_2', 'M18_3', 'M18_4', 'M19_1', 'M19_2', 'M19_3', 'M19_4',
       'M20_1', 'M20_2', 'M20_3', 'M20_4', 'M21_1', 'M21_2', 'M21_3', 'M22_1', 'M22_2', 'M22_3', 'M22_4',
       'M23_1', 'M23_2', 'M23_3', 'M23_4', 'M24_1', 'M24_2', 'M24_3', 'M24_4','M25_1','M25_2', 'M25_3', 'M25_4',
       'M26_1','M26_2', 'M26_3', 'M26_4', 'M27_1','M27_2', 'M27_3', 'M27_4', 'M28_1','M28_2', 'M28_3', 'M28_4',
       'M29_1', 'M29_2', 'M29_3', 'M29_4', 'M30_1', 'M30_2', 'M30_3', 'M30_4',
       'label']
motif_f.write(','.join(col) + '\n')
node_data = node_hum[:2475] + node_rob
for i in (node_data):
        t = users_label[i]
        num_M1 = M1(G, i)
        num_M2 = M2(G, i)
        num_M3 = M3(G, i)
        num_M4 = M4(G, i)
        num_M5 = M5(G, i)
        num_M6 = M6(G, i)
        num_M7 = M7(G, i)
        num_M8 = M8(G, i)
        num_M9 = M9(G, i)
        num_M10 = M10(G, i)
        num_M11 = M11(G, i)
        num_M12 = M12(G, i)
        num_M13 = M13(G, i)
        num_M14 = M14(G, i)
        num_M15 = M15(G, i)
        num_M16 = M16(G, i)
        num_M17 = M17(G, i)
        num_M18 = M18(G, i)
        num_M19 = M19(G, i)
        num_M20 = M20(G, i)
        num_M21 = M21(G, i)
        num_M22 = M22(G, i)
        num_M23 = M23(G, i)
        num_M24 = M24(G, i)
        num_M25 = M25(G, i)
        num_M26 = M26(G, i)
        num_M27 = M27(G, i)
        num_M28 = M28(G, i)
        num_M29 = M29(G, i)
        num_M30 = M30(G, i)
        num_M = (num_M1 + num_M2 + num_M3 + num_M4 + num_M5 + num_M6 + num_M7 + num_M8 + num_M9 + num_M10 + num_M11 + num_M12
        + num_M13 + num_M14 + num_M15 + num_M16 + num_M17 + num_M18 + num_M19 + num_M20 + num_M21 + num_M22 + num_M23 + num_M24
        + num_M25 + num_M26 + num_M27 + num_M28 + num_M29 + num_M30)
        motif_f.write(i+','+','.join(num_M)+','+str(t)+'\n')
motif_f.close()





