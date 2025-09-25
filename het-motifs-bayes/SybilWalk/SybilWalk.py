from copy import copy

import networkx as nx
import pandas as pd

f = open(r'cresci15-network.txt')
net = []
for i in f:
    t = i.strip('\n').split(' ')
    net.append([t[0], t[1], 1])
G = nx.DiGraph()
G.add_weighted_edges_from(net)
node_list = list(G.nodes)

edge_list = list(G.edges)
print("node num ",len(node_list),"    edge num ",len(edge_list))


#赋予标签
f_label = open(r'cresci15-label.txt')
label = {}
for i in f_label:
    t = i.strip('\n').split(' ')
    label[t[0]]=t[1]
print('len users_label:',len(label))

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


node_hum = node_hum[:650]  #cresci_15参数为650，MGTAB参数为2475



#3. 划分训练集和测试集,cresci-15网络的参数为571，MGTAB的参数为2227
train_data = node_hum[:571] + node_rob[:571]
test_data = node_hum[571:] + node_rob[571:]

#添加一定比例噪声
df_noise = pd.read_csv(r'MGTAB_motif_noise_50%.csv')
node_noise = df_noise['src'].tolist()
label_noise = df_noise['label'].tolist()
for i in range(len(node_noise)):
    label[str(node_noise[i])] = str(label_noise[i])


#4. sybilwalk
#构建标签增强网络,0-人类，1-机器人

for i in train_data:
        if label[i] == '0':
            G.add_weighted_edges_from([('lb', i, 1)])

        else:
            G.add_weighted_edges_from([('ls', i, 1)])

#检测算法
thres=10**(-100)
T=20  #30
score={'lb':0,'ls':1}
for i in G.nodes:
    if i!='ls' and i!='lb':
        score[i]=0.5
score_s = copy(score)
#第一次迭代算出每个结点的评分
for user in G.nodes:
        if user != 'lb' and user != 'ls':
            neighbor = G[user]
            p1 = 0
            for j in neighbor.keys():
                p1 = p1 + neighbor[j]['weight']
            p2 = 0
            for m in neighbor.keys():
                p2 = p2 + (neighbor[m]['weight'] / p1) *score_s[m]
            score[user] = p2
t=2

#计算连续两次迭代中所有用户节点的不良评分变化是否大于等于阈值
def ptotal(x,x_s):
    n=0
    sum=0
    for i in x.keys():
        if (x[i]-x_s[i])**2>=10**(-350):
            n=n+1
            sum=(sum+x[i]-x_s[i])/n
    print('第n次迭代的误差为：',abs(sum))
    '''
    if n==len(x):
        return True
    else:
        return False
        '''

#ptotal(score,score_s)
while t<=T:
    ptotal(score, score_s)
    score_s=copy(score)
    for i in G.nodes:
        if   i!='lb' and i!='ls':
            neighbor1=G[i]
            p3=0
            for j in neighbor1.keys():
                p3=p3+neighbor1[j]['weight']
            p4=0
            for m in neighbor1.keys():
                p4=p4+(neighbor1[m]['weight']/p3)*score_s[m]
            score[i]=p4
    t=t+1

y_true=[]
y_pred=[]
for i in test_data:
    if label[i]== '0':
        y_true.append(0)
    else:
        y_true.append(1)
    if score[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, f1_score, precision_score, recall_score
y_true=np.array(y_true)
y_pred=np.array(y_pred)
print(accuracy_score(y_true,y_pred,normalize=False))#预测正确的个数
print(accuracy_score(y_true,y_pred))#准确率
print(confusion_matrix(y_true,y_pred))##混淆矩阵
from sklearn.metrics import classification_report

target = ['human', 'bot']
print(classification_report(y_true, y_pred, target_names=target))
print(classification_report(y_true, y_pred, target_names=target))
print("ACC: {}".format(accuracy_score(y_true, y_pred)))
print("AUC: {}".format(roc_auc_score(y_true, y_pred)))
print("F1: {}".format(f1_score(y_true, y_pred)))
print("Precision: {}".format(precision_score(y_true, y_pred)))
print("Recall: {}\n".format(recall_score(y_true, y_pred)))

