#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd

# 数据导入
data = pd.read_excel('D:/HBU/研一/2 研一下/07 统计预测与决策/课程论文/Kickstarter-3.8-kickstarter_all_datatrain.xlsx')

# 将缺失值填充为0
data.fillna(0, inplace=True)

# 将Success中的字符串改为数字0，1
keep1 = 'successful'
keep2 = 'failed'
data.loc[~data['Success'].isin([keep1, keep2]), 'Success'] = pd.NA
data['Success'] = data['Success'].replace({'successful': 1, 'failed': 0})

# 将Video中的除0外的字符串改为1
data.loc[data['Video'] != 0, 'Video'] = 1

# 将多个字符变量转换为数值型变量
data[['Success', 'Video']] = data[['Success', 'Video']].apply(pd.to_numeric, errors='coerce')

# 删除Success中包含缺失值的行
data_dropna = data.dropna(subset=['Success'])

# 删除包含缺失值的行
data = data.dropna()

# 设置浮点数格式，保留三位小数
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
print(data.describe())

import datetime
import re

# 定义一个函数将字符串转换为秒数
def time_to_seconds(time_str):
    # 定义正则表达式来匹配不同格式的时间字符串
    time_regex = re.compile(r'(?:(\d+)天)?(?:(\d+)小时)?(?:(\d+)分钟)?(?:(\d+)秒)?')
    # 使用正则表达式匹配字符串中的不同部分
    time_parts = time_regex.match(time_str).groups()
    # 将匹配的结果转换为整数，并用 0 填充缺失部分
    days = int(time_parts[0]) if time_parts[0] else 0
    hours = int(time_parts[1]) if time_parts[1] else 0
    minutes = int(time_parts[2]) if time_parts[2] else 0
    seconds = int(time_parts[3]) if time_parts[3] else 0
    # 使用 timedelta 函数计算总秒数
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds

# 将 data['Duration'] 列中的所有字符串转换为秒数
data['Duration_seconds'] = data['Duration'].apply(time_to_seconds)

# 仅对 'Duration_seconds' 变量进行描述性统计
Duration_stats = data['Duration_seconds'].describe()
print(Duration_stats)

# 将秒转换为天
data['Duration_seconds'] = data['Duration_seconds'] / (24 * 60 * 60)

# 将新的变量重命名为 'days'
data = data.rename(columns={'Duration_seconds': 'days'})

data = data.drop(['title', 'Start', 'Close', 'Duration'], axis=1)

data.info()


# In[109]:


data = data.drop(['region', 'Category'], axis=1)


# In[111]:


data = data.drop(['Backers', 'CommentNum', 'days'], axis=1)


# In[112]:


# 对'region'变量进行one-hot编码
data_encodedregion = pd.get_dummies(data, columns=['region'])

# 对'Category'变量进行one-hot编码
data_encodedCategory = pd.get_dummies(data, columns=['Category'])

# 合并编码后的数据
data_encoded = pd.concat([data, data_encodedregion, data_encodedCategory], axis=1)

# 删除原始的'region'和'Category'变量
data_encoded.drop(['region', 'Category'], axis=1, inplace=True)

data_encoded.head()


# In[19]:


# 保存为excel
data.to_excel('processed_data.xlsx', index=False)


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# In[20]:


# 读取数据集
data = pd.read_excel('D:/Python/Scripts/LiuqianCourse/processed_data.xlsx')


# In[114]:


# 提取自变量和因变量
X = data_encoded.drop('Success', axis=1)
y = data_encoded['Success']

# 将数据集划分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器并训练模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 对测试数据进行预测并计算准确率
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[101]:


# 对'region'变量进行one-hot编码
data_encodedregion = pd.get_dummies(data, columns=['region'])

# 对'Category'变量进行one-hot编码
data_encodedCategory = pd.get_dummies(data, columns=['Category'])

# 对'Success'变量进行one-hot编码
data_encodedSuccess = pd.get_dummies(data, columns=['Success'])

# 合并编码后的数据
data_encoded = pd.concat([data, data_encodedregion, data_encodedCategory, data_encodedSuccess], axis=1)

# 删除原始的'region'和'Category'变量
data_encoded.drop(['region', 'Category', 'Success'], axis=1, inplace=True)


# In[104]:


data_encoded.head()


# In[105]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data_encoded.drop('Success_1.0', axis=1)
y = data_encoded['Success_1.0']


# In[ ]:


print(df.dtypes)


# In[106]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 划分训练集和测试集
lr = LogisticRegression(max_iter=1000, random_state=42)  # 创建 LR 模型
lr.fit(X_train, y_train)  # 训练模型
y_pred = lr.predict(X_test)  # 预测标签
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy:', accuracy_score(y_test, y_pred))


# In[ ]:




