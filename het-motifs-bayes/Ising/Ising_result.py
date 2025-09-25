import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score




df=pd.read_csv(r'MGTAB_tong_motif.csv')
data=list(df['src'])
y_true=list(df['label'])

df_pred=pd.read_csv('MGTAB_result.csv')
data_pred=list(df_pred['src'])
data_label=list(df_pred['label'])
node_label={}
for i in range(len(data_pred)):
    node_label[data_pred[i]]=data_label[i]

y_pred=[]
for i in range(len(data)):
    y_pred.append(int(node_label[data[i]]))
acc=accuracy_score(y_true, y_pred)
auc=roc_auc_score(y_true, y_pred)
f1=f1_score(y_true, y_pred)
precision=precision_score(y_true, y_pred)
recall=recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)
print('ACC:',acc)
print('precision:', precision)
print('Recall:', recall)
print('f1:',f1)
print('auc:',auc)