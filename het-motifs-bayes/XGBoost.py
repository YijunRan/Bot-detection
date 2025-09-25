import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

df = pd.read_csv('TwiBot22_yi_motif_feature.csv')

# 选取up_bound_ml>0.7的特征进行10折交叉验证
df_bound = pd.read_csv('TwiBot22_up_bound_ml_result.csv')
title_bound_score_bigger_70 = df_bound[df_bound['ub_bound_ml']>=0.7]['features'].to_list()
print('features of up_bound_ml > 0.7:', len(title_bound_score_bigger_70))
print(title_bound_score_bigger_70)
# df = df[title_bound_score_bigger_70]

# columns = list(df.columns)[1:-1]

# #选取up_bound_ml>0.7的特征进行10折交叉验证
columns = title_bound_score_bigger_70

print(columns)
Xtrain_all = df[columns]
Ytrain = df['label']




# 构建随机森林分类器
Xtrain = pd.DataFrame(Xtrain_all.values)
print(Xtrain.shape)
# 使用十折交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 初始化列表以存储指标值
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
AUC=[]

# 交叉验证并计算指标
i=0
for train_index, val_index in skf.split(Xtrain, Ytrain):
    X_train_fold, X_val_fold = Xtrain.iloc[train_index], Xtrain.iloc[val_index]
    y_train_fold, y_val_fold = Ytrain.iloc[train_index], Ytrain.iloc[val_index]

    auc_max, acc_max, precision_max, recall_max, f1_max = 0, 0, 0, 0, 0
    n_est = [50, 100, 150]
    m_depth = [5, 15, 25, 45]
    auc_max, acc_max, precision_max, recall_max, f1_max = 0, 0, 0, 0, 0
    for ii in n_est:
        for jj in m_depth:
            print(f"当前参数n_est{ii}，m_depth{jj}，第{i}折")
            clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=ii, max_depth=jj)
            clf.fit(X_train_fold, y_train_fold)
            clf_proba = clf.predict_proba(X_val_fold)

            y_pred = clf.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_pred)
            precision = precision_score(y_val_fold, y_pred, average='binary')  # 或者使用'macro', 'micro', 'weighted'等
            recall = recall_score(y_val_fold, y_pred, average='binary')
            f1 = f1_score(y_val_fold, y_pred, average='binary')
            auc = roc_auc_score(y_val_fold, clf_proba[:, 1])
            if auc > auc_max:
                auc_max = auc
                acc_max = accuracy
                precision_max = precision
                recall_max = recall
                f1_max = f1
            results = {}
            results['accuracy'] = acc_max
            results['precision'] = precision_max
            results['recall'] = recall_max
            results['f1'] = f1_max
            results['roc_auc'] = auc_max

            # 计算各项指标的均值和标准差
            metrics_summary = {}
            for metric in results:
                values = np.array(results[metric])
                metrics_summary[f'{metric}_mean'] = np.mean(values)
                metrics_summary[f'{metric}_std'] = np.std(values)

    # # 打印出每折的mean&std
    # print('第'+str(i)+'次验证：')
    # print("随机森林模型十折交叉验证结果：")
    # for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    #     print(
    #         f"{metric.upper()}: 均值 = {metrics_summary[f'{metric}_mean']:.4f}, 标准差 = {metrics_summary[f'{metric}_std']:.4f}")

    i +=1
    AUC.append(auc_max)
    accuracy_scores.append(acc_max)
    precision_scores.append(precision_max)
    recall_scores.append(recall_max)
    f1_scores.append(f1_max)


# 输出结果
for i in range(10):
    print(f"第{i}次交叉验证结果：")
    print(f"Accuracy: {accuracy_scores[i]:.4f}")
    print(f"Precision: {precision_scores[i]:.4f}")
    print(f"Recall: {recall_scores[i]:.4f}")
    print(f"F1-score: {f1_scores[i]:.4f}")
    print(f"AUC: {AUC[i]:.4f}")

# 查看所有迭代的平均值
print(f"Average Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f}")
print(f"Average Precision: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"Average Recall: {sum(recall_scores) / len(recall_scores):.4f}")
print(f"Average F1-score: {sum(f1_scores) / len(f1_scores):.4f}")
print(f"Average AUC: {sum(AUC) / len(AUC):.4f}")

df = pd.DataFrame(
pd.DataFrame({
    "Accuracy": accuracy_scores,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1": f1_scores,
    "AUC": AUC,
})
)
df.to_csv('TwiBot22_yi_selected_params_features.csv', index=False, encoding='utf-8-sig')
