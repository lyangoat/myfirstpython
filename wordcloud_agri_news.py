#!/usr/bin/env python
# coding: utf-8

# In[11]:


import jieba
import pandas as pd
import numpy as np
import cv2
from wordcloud import WordCloud
import PIL.Image as image


# In[13]:


df=pd.read_excel('D:/HBU/1/1_2/11survey/stat/modify1/data绿色食品.xls')
df=df[['标题','新闻内容']]
df.shape


# In[14]:


df = df.loc[df['标题'].str.contains('农')]
df.shape


# In[15]:


df['text'] = df['标题'] + ' ' + df['新闻内容']
df['text'].to_csv('agri_news.txt', index=False, header=False)


# In[16]:


def cut(text):
    word_list=jieba.cut(text,cut_all=False)
    result=' '.join(word_list)
    return result


# In[17]:


text = open("agri_news.txt", "r", encoding="utf-8").read()
seg_list=jieba.cut(text,cut_all=False)
print('/'.join(seg_list))


# In[18]:


stopwords = {}.fromkeys(['的', '了','我','是','吗','吃','有','没有',
                         '回答','不','就','日报','晚报','新闻','转载','本报',
                        '据','通讯员','声明','报告','晚报','和','等',
                         '为','个','以','目前','与','对','和','在','在于'])
#添加禁用词之后
seg_list = jieba.cut(text)
final = ''
for seg in seg_list:
    if seg not in stopwords:
        final += seg
seg_list_new = jieba.cut(final)
print(u"[切割之后]: ", "/ ".join(seg_list_new))


# In[19]:


with open('agri_news.txt',encoding="utf-8") as f:
    text=f.read()
    text=cut(final)
mask = np.array(cv2.imread('D:/HBU/1/1_2/11survey/stat/modify1/P-1543472-66EF71EAO.jpg'))
print(mask.shape)


# In[20]:


wordcloud=WordCloud(mask=mask,colormap='summer',background_color='#FFFFFF',font_path='C:/Windows/Fonts/simhei.ttf').generate(text)
image_produce=wordcloud.to_image()
image_produce.show()


# In[ ]:




