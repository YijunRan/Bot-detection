from copy import copy

import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

#1.构建网络
f = open(r'MGTAB_edge_index_followers_friendes.txt')
net = []
for i in f:
    t = i.strip('\n').split(',')
    net.append([t[0], t[1], 1])
G = nx.DiGraph()
G.add_weighted_edges_from(net)
node_list = list(G.nodes)

edge_list = list(G.edges)
print("node num ",len(node_list),"    edge num ",len(edge_list))



#赋予标签
f_label = open(r'MGTAB_label.txt')
label = {}
for i in f_label:
    t = i.strip('\n').split(',')
    label[t[0]]=t[1]
print('len users_label:',len(label))


df_noise = pd.read_csv(r'MGTAB_motif_noise_5%.csv')
node_noise = df_noise['src'].tolist()
label_noise = df_noise['label'].tolist()
for i in range(len(node_noise)):
    label[str(node_noise[i])] = str(label_noise[i])

c_rob = 0 #计算网络中机器人节点个数
c_hum = 0
node_hum = []
node_rob = []
for i in label.keys():
    if (label[i] == '1') and (i in node_list):
        node_rob.append(i)
        c_rob += 1
    elif (label[i] == '0') and (i in node_list):
        node_hum.append(i)
        c_hum += 1
print('网络中机器人节点个数：',c_rob)
print('网络中人类节点个数：',c_hum)

node_hum = node_hum[:2475]



#3. 划分训练集和测试集
train_data = node_hum[:2227] + node_rob[:2227]
test_data = node_hum[2227:] + node_rob[2227:]
print(train_data)

#4.检测算法
theta,T=0.1,20
thres=10**(-3)
p={}
q={}
score={}
for i in node_list:
        if label[i]== '0' and (i not in test_data):
            q[i]=0.5-theta-0.5
        elif label[i]== '1' and (i not in test_data):
            q[i]=0.5+theta-0.5
for i in test_data:
    q[i]=0.5-0.5
p_s=copy(q)
t=1
#计算连续两次迭代中所有用户节点的不良评分变化是否大于等于阈值
def ptotal(p2,p1):
    max=p['0']
    for i in p1.keys():
        t=abs(p2[i]-p1[i])
        if t>max:
            max=t
    max0=p['0']
    for i in p2.keys():
        if p2[i]>max0:
            max0=p2[i]
    return max/max0
#ptotal(p,p_s)>=thres and t<T
while  t<T:
    for user in node_list:
            neighbor = G[user]
            p2=0
            for m in neighbor.keys():
                p2 = p2+0.0001*p_s[m]
            p2=2*p2+q[user]
            p[user] = p2
    wu=ptotal(p, p_s)
    print('第{}次迭代误差为{}'.format(t,wu))
    p_s = copy(p)
    t=t+1
for i in p.keys():
    p[i] = p[i] + 0.5

y_true=[]
y_pred=[]
for i in test_data:
    if label[i]== '0':
        y_true.append(0)
    else:
        y_true.append(1)
    if p[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
y_true=np.array(y_true)
y_pred=np.array(y_pred)
print(accuracy_score(y_true,y_pred,normalize=False))#预测正确的个数
print(accuracy_score(y_true,y_pred))#准确率
print(confusion_matrix(y_true,y_pred))##混淆矩阵
from sklearn.metrics import classification_report

target = ['human', 'bot']
print(classification_report(y_true, y_pred, target_names=target))
print("ACC: {}".format(accuracy_score(y_true, y_pred)))
print("AUC: {}".format(roc_auc_score(y_true, y_pred)))
print("F1: {}".format(f1_score(y_true, y_pred)))
print("Precision: {}".format(precision_score(y_true, y_pred)))
print("Recall: {}\n".format(recall_score(y_true, y_pred)))
