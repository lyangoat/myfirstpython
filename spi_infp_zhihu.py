#!/usr/bin/env python
# coding: utf-8

# In[96]:


# 导包
import os
import pandas as pd
from selenium import webdriver
from lxml import etree
import time
import jieba
import re
import numpy as np

from selenium.webdriver.chrome.service import Service

# 创建一个Service对象，参数为driver的路径
s = Service("D:/Python/chromedriver.exe")

# 使用这个Service对象来创建WebDriver
browser = webdriver.Chrome(service=s)
url = 'https://www.zhihu.com/question/446820604/answer/2534982126'

# 直接得到所有答案
parts = url.split('/')  # 分割字符串
url = '/'.join(parts[:5])  # 只保留前面的部分
browser.get(url)

# 关闭登录窗
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

wait = WebDriverWait(browser, 10)
close_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='关闭']")))
close_button.click()


# In[68]:


# 下滑到底
for i in range(20):
    browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(0.5)
    print(i)


# In[78]:


import time

# 设置滚动等待时间
SCROLL_PAUSE_TIME = 1

# 获取当前页面的高度
last_height = browser.execute_script("return document.body.scrollHeight")

# 记录连续无变化的次数
no_change_count = 0

while no_change_count < 5:
    # 向下滚动到页面底部
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 等待页面加载
    time.sleep(SCROLL_PAUSE_TIME)

    # 计算新的页面高度并与上一次的页面高度进行比较
    new_height = browser.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        no_change_count += 1
        if no_change_count == 5:
            # 如果连续5次页面没有变化，则退出循环
            break
    else:
        # 重置连续无变化的计数
        no_change_count = 0

    # 向上滚动1000像素
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight - 1000);")
    time.sleep(SCROLL_PAUSE_TIME)

    # 更新最后的页面高度
    last_height = browser.execute_script("return document.body.scrollHeight")


# In[97]:


from selenium.webdriver.common.by import By
final_end_it = browser.find_elements(By.XPATH, """//button[contains(@class,"Button") 
and contains(@class ,'QuestionAnswers-answerButton')
and contains(@class ,'Button--blue')
and contains(@class ,'Button--spread')]""")

# 爬取
while final_end_it == []:
    final_end_it = browser.find_elements(By.XPATH, """//button[contains(@class,"Button") 
and contains(@class ,'QuestionAnswers-answerButton')
and contains(@class ,'Button--blue')

and contains(@class ,'Button--spread')
]""")
    js="var q=document.documentElement.scrollTop=0"  
    browser.execute_script(js)
    for i in range(30):
        browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(0.5)
        print(i)

dom = etree.HTML(browser.page_source)
# 问题本身的数据
Followers_number_first = dom.xpath("""//div[@class="QuestionFollowStatus"]//div[@class = "NumberBoard-itemInner"]/strong/text()""")[0]
Browsed_number_first = dom.xpath("""//div[@class="QuestionFollowStatus"]//div[@class = "NumberBoard-itemInner"]/strong/text()""")[1]

# 关注者数量
Followers_number_final = re.sub(",","",Followers_number_first)

# 浏览数量
Browsed_number_final = re.sub(",","",Browsed_number_first)

# 问题链接
problem_url =  url1

# 问题ID
problem_id = re.findall(r"\d+\.?\d*",url1)

# 问题标题
problem_title =  dom.xpath("""//div[@class = 'QuestionHeader']//h1[@class = "QuestionHeader-title"]/text()""")

# 问题点赞数
problem_endorse = dom.xpath("""//div[@class = 'QuestionHeader']//div[@class = "GoodQuestionAction"]/button/text()""")

# 问题评论数
problem_Comment = dom.xpath("""//div[@class = 'QuestionHeader']//div[@class = "QuestionHeader-Comment"]/button/text()""")

# 问题回答数
answer_number = dom.xpath("""//div[@class = 'Question-main']//h4[@class = "List-headerText"]/span/text()""")

# 问题标签
problem_tags_list = dom.xpath("""//div[@class = 'QuestionHeader-topics']//a[@class = "TopicLink"]/div/div/text()""")

# 答案具体内容
comment_list = dom.xpath("""//div[@class = 'List-item']//div[@class = "RichContent-inner"]""")
comment_list_text = []
for comment in comment_list:
    comment_list_text.append(comment.xpath("string(.)"))
    
# 发表时间
time_list = dom.xpath("""//div[@class = 'List-item']//div[@class = "ContentItem-time"]//span/@data-tooltip""")
edit_time_list = dom.xpath("""//div[@class = 'List-item']//div[@class = "ContentItem-time"]//span/text()""")

# 点赞数
endorse_list = dom.xpath("""//div[@class = 'List-item']//button[contains(@class,"Button") and contains(@class,"VoteButton") and contains(@class , "VoteButton--up")]/@aria-label""")

# 评论人数
number_of_endorse_list = dom.xpath("""//div[@class = 'List-item']//svg[contains(@class,"Zi")   and contains(@class,"Zi--Comment") 
and contains(@class,"Button-zi")]/../../text()""")

# 回答链接
answers_url_list = dom.xpath("""//div[@class = 'List-item']//div[contains(@class,"ContentItem") and contains(@class,"AnswerItem")]
/meta[@itemprop = "url"]/@content""")
authors_list = dom.xpath("""//div[@class = 'List-item']//div[contains(@class,"ContentItem") and contains(@class,"AnswerItem")]
/@data-zop""")

# 作者姓名
authorName_list = []

# 作者id
authorid_list = []
for i in authors_list:
    authorName_list.append(eval(i)['authorName'])
    authorid_list.append(eval(i)["itemId"])


# 合成数据框
data = pd.DataFrame()

data['具体内容'] = comment_list_text
data["发表时间"] = time_list
data["点赞数"] = endorse_list
data["评论人数"] = number_of_endorse_list
data["回答链接"] = answers_url_list
data["作者姓名"]  = authorName_list
data['作者id'] = authorid_list


data["关注者数量"] = Followers_number_final
data["浏览数量"] = Browsed_number_final
data["问题链接"] = problem_url
data["问题ID"]  = problem_id[0]
data["问题标题"] = problem_title[0]
data["问题点赞数"] = problem_endorse[0]
data["问题评论数"] = problem_Comment[0]
data["问题回答数"] = answer_number[0]
data["问题标签"] = "&".join(problem_tags_list)


# In[98]:


data.head()


# In[101]:


data.to_excel('D:/HBU/1/1_2/11survey/stat/modify1/datain.xlsx', index=False)


# In[ ]:




