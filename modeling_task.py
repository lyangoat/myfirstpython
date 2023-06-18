#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# 数据导入
data = pd.read_excel('D:/HBU/研一/2 研一下/11 市调/统计建模/task1/5.19 随机森林数据 (2).xlsx', sheet_name= 'stata')

data.info()


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 提取自变量和因变量
X = data.drop('purc', axis=1)
y = data['purc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 划分训练集和测试集
lr = LogisticRegression(max_iter=1000, random_state=42)  # 创建 LR 模型
lr.fit(X_train, y_train)  # 训练模型
y_pred = lr.predict(X_test)  # 预测标签
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:', accuracy_score(y_test, y_pred))


# In[16]:


import statsmodels.api as sm

# 添加截距列
X_train = sm.add_constant(X_train)

# 创建并训练Logit模型
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# 打印显著性表
print(result.summary())


# In[17]:


# 删除不显著变量
data_lr = data.drop(['year', 'pnum', 'resid', 'income'], axis=1)


# In[18]:


data_lr.info()


# In[19]:


# 对剔除不显著变量后的数据进行逻辑回归

# 提取自变量和因变量
X = data_lr.drop('purc', axis=1)
y = data_lr['purc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 划分训练集和测试集
lr = LogisticRegression(max_iter=1000, random_state=42)  # 创建 LR 模型
lr.fit(X_train, y_train)  # 训练模型
lr_y_pred = lr.predict(X_test)  # 预测标签
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:', accuracy_score(y_test, lr_y_pred))

# 添加截距列
X_train = sm.add_constant(X_train)

# 创建并训练Logit模型
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# 打印显著性表
print(result.summary())


# In[20]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 获取预测为正例的概率
lr_y_pred_prob = lr.predict_proba(X_test)[:, 1]

# 计算FPR、TPR和阈值
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_y_pred_prob)
lr_roc_auc = auc(lr_fpr, lr_tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkorange', label='ROC curve (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[21]:


### 做一下对比

import pandas as pd

# 数据导入
data_1 = pd.read_csv('D:/HBU/研一/2 研一下/11 市调/统计建模/task1/5.20 逻辑回归data.csv')

data_1.info()


# In[22]:


# 提取自变量和因变量
X = data_1.drop('purc', axis=1)
y = data_1['purc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 划分训练集和测试集
lr = LogisticRegression(max_iter=1000, random_state=42)  # 创建 LR 模型
lr.fit(X_train, y_train)  # 训练模型
y_pred = lr.predict(X_test)  # 预测标签
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:', accuracy_score(y_test, y_pred))

# 添加截距列
X_train = sm.add_constant(X_train)

# 创建并训练Logit模型
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# 打印显著性表
print(result.summary())


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 提取自变量和因变量
X = data.drop('purc', axis=1)
y = data['purc']

# 将数据集划分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器并训练模型
rf = RandomForestClassifier(random_state = 4, warm_start = True, 
                                n_estimators = 300, oob_score = True,
                                max_depth = 4, max_features = 'sqrt')
rf.fit(X_train, y_train)

# 对测试数据进行预测并计算准确率
rf_y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, rf_y_pred)
print('Accuracy:', accuracy)

# 获取特征重要性
importance = rf.feature_importances_
features = X.columns

# 特征重要性排序
indices = np.argsort(importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# 预测概率
rf_y_prob = rf.predict_proba(X_test)[:, 1]

# 计算TPR和FPR
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_y_prob)

# 计算AUC
rf_roc_auc = auc(rf_fpr, rf_tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(rf_fpr, rf_tpr, label='ROC curve (AUC = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[12]:


# 绘制两者ROC曲线
plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkorange', lw=2, label='Logistic Regression (AUC = %0.2f)' % lr_roc_auc)
plt.plot(rf_fpr, rf_tpr, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[72]:


# 参数选择

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 提取自变量和因变量
X = data.drop('purc', axis=1)
y = data['purc']

# 将数据集划分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'random_state': [4, 42, 62],
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

# 创建随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合和对应的准确率
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# 使用最佳参数训练模型并进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# In[26]:


from sklearn.metrics import confusion_matrix, classification_report

# 对随机森林模型计算混淆矩阵
rf_cm = confusion_matrix(y_test, rf_y_pred)

# 对逻辑回归模型计算混淆矩阵
lr_cm = confusion_matrix(y_test, lr_y_pred)

# 计算随机森林模型的特异度、敏感度、准确性、阳性预测值和阴性预测值
rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()
rf_specificity = rf_tn / (rf_tn + rf_fp)
rf_sensitivity = rf_tp / (rf_tp + rf_fn)
rf_accuracy = (rf_tp + rf_tn) / (rf_tp + rf_tn + rf_fp + rf_fn)
rf_precision = rf_tp / (rf_tp + rf_fp)
rf_negative_predictive_value = rf_tn / (rf_tn + rf_fn)

# 计算逻辑回归模型的特异度、敏感度、准确性、阳性预测值和阴性预测值
lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()
lr_specificity = lr_tn / (lr_tn + lr_fp)
lr_sensitivity = lr_tp / (lr_tp + lr_fn)
lr_accuracy = (lr_tp + lr_tn) / (lr_tp + lr_tn + lr_fp + lr_fn)
lr_precision = lr_tp / (lr_tp + lr_fp)
lr_negative_predictive_value = lr_tn / (lr_tn + lr_fn)

# 打印结果
print("随机森林模型：")
print("特异度：", rf_specificity)
print("敏感度：", rf_sensitivity)
print("准确性：", rf_accuracy)
print("阳性预测值：", rf_precision)
print("阴性预测值：", rf_negative_predictive_value)

print("逻辑回归模型：")
print("特异度：", lr_specificity)
print("敏感度：", lr_sensitivity)
print("准确性：", lr_accuracy)
print("阳性预测值：", lr_precision)
print("阴性预测值：", lr_negative_predictive_value)


# In[ ]:




