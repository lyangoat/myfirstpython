#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import jieba
import jieba.posseg as pseg
from collections import Counter

# 从Excel文件中读取评论内容
data = pd.read_excel('D:/HBU/1/1_2/11survey/stat/modify1/datain.xlsx')
comments = data['具体内容'].tolist()

# 分词和词性标注，筛选形容词进行统计
word_counter = Counter()

for comment in comments:
    # 将评论内容转换为字符串类型
    comment = str(comment)
    
    # 使用jieba进行中文分词和词性标注
    words = pseg.cut(comment)
    
    # 筛选形容词并排除长度为1的形容词和特定词进行统计
    adj_words = [word.word for word in words if word.flag.startswith('a') and len(word.word) > 1 
                 and word.word not in ['容易', '不同', '完全', '突然', '一般', '直接', '确实'
                                       ,'最好', '根本','很大', '充满']]
    
    # 统计形容词词频
    word_counter.update(adj_words)

# 获取词频最高的前30个形容词
top_words = word_counter.most_common(30)

# 打印词频最高的30个形容词
for word, count in top_words:
    print(word, count)


# In[ ]:




