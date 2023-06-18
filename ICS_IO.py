#!/usr/bin/env python
# coding: utf-8

# In[27]:


# 用来添加IO表空单元格

import pandas as pd
from openpyxl import load_workbook

# 打开Excel文件
filename = 'D:/HBU/1/1_2/01 国民经济核算与投入产出分析/JAPAN.xlsx'
book = load_workbook(filename)

# 遍历要计算的表格，填充3个空白单元格
for i in range(2, 16):
    # 构造表格名称
    sheet_name = f'Table 1.{i}'
    fixed_text = 'c'  # 要输入的固定字符

    # 选择要操作的表格
    sheet = book[sheet_name]
    
    # 输入固定的字符到指定的单元格
    sheet['B6'] = fixed_text
    
for i in range(2, 16):
    # 构造表格名称
    sheet_name = f'Table 1.{i}'
    fixed_text = 'gva'  # 要输入的固定字符

    # 选择要操作的表格
    sheet = book[sheet_name]
    
    # 输入固定的字符到指定的单元格
    sheet['B50'] = fixed_text
    
for i in range(2, 16):
    # 构造表格名称
    sheet_name = f'Table 1.{i}'
    fixed_text = 'total'  # 要输入的固定字符

    # 选择要操作的表格
    sheet = book[sheet_name]
    
    # 输入固定的字符到指定的单元格
    sheet['AR6'] = fixed_text    

# 保存修改后的Excel文件
book.save('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/JAPAN.xlsx')


# In[25]:


# 中国ICS计算

import pandas as pd
import matplotlib.pyplot as plt

# 创建一个空列表，用于存储ICS值和相关数据
data = []

# 遍历要计算的表格
for i in range(2, 16):
    # 构造表格名称
    sheet_name = f'Table 1.{i}'
    
    # 读取Excel文件中的特定表格，跳过前五行
    df = pd.read_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/CHINA.xlsx', sheet_name=sheet_name, skiprows=5)
    
    # 剔除第一列数据
    df = df.drop(df.columns[0], axis=1)
    
    # 选择需要相加的列
    columns_to_sum = ['c8', 'c9', 'c11', 'c12', 'c13', 'c14', 'c15']
    
    # 相加得到新列'chain'
    df['chain'] = df[columns_to_sum].sum(axis=1)
    
    # 将第一列设置为行标签
    df = df.set_index(df.columns[0])
    
    # 获取chain列和gva行对应的数据
    va = df.at['gva', 'chain']
    
    # 获取chain列和r69行对应的数据
    to = df.at['r69', 'chain']
    
    # 计算ICS值，并保留4位小数
    ics = round(va / to, 4)
    
    # 将相关数据添加到列表中
    data.append({'Table Number': i, 'va': round(va, 2), 'to': round(to, 2), 'ics': ics})

# 创建DataFrame
df_data = pd.DataFrame(data)

# 打印数据框
print(df_data)

# 保存数据框为Excel文件
df_data.to_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/ICS_Values.xlsx', index=False)

# 绘制折线图

# 创建新的图表
plt.figure(figsize=(10, 6))

plt.plot(df_data['Table Number'], df_data['ics'], marker='o', linestyle='-', label='ICS Value')
plt.xlabel('Year')
plt.ylabel('ICS Value')
plt.title('ICS Values for 2007 to 2020')
plt.ylim(0, 0.3)
plt.xticks(range(2, 16), range(2007, 2021))

# 设置刻度标签的旋转角度为45度
plt.xticks(range(2, 16), range(2007, 2021), rotation=45)

plt.legend()

# 提高输出图像的分辨率
plt.savefig('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/ICS_Values_Plot.png', dpi=300)

plt.show()


# In[29]:


# 日本ICS计算

import pandas as pd
import matplotlib.pyplot as plt

# 创建一个空列表，用于存储ICS值和相关数据
data = []

# 遍历要计算的表格
for i in range(2, 16):
    # 构造表格名称
    sheet_name = f'Table 1.{i}'
    
    # 读取Excel文件中的特定表格，跳过前五行
    df = pd.read_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/JAPAN.xlsx', sheet_name=sheet_name, skiprows=5)
    
    # 剔除第一列数据
    df = df.drop(df.columns[0], axis=1)
    
    # 选择需要相加的列
    columns_to_sum = ['c8', 'c9', 'c11', 'c12', 'c13', 'c14', 'c15']
    
    # 相加得到新列'chain'
    df['chain'] = df[columns_to_sum].sum(axis=1)
    
    # 将第一列设置为行标签
    df = df.set_index(df.columns[0])
    
    # 获取chain列和gva行对应的数据
    va = df.at['gva', 'chain']
    
    # 获取chain列和r69行对应的数据
    to = df.at['r69', 'chain']
    
    # 计算ICS值，并保留4位小数
    ics = round(va / to, 4)
    
    # 将相关数据添加到列表中
    data.append({'Table Number': i, 'va': round(va, 2), 'to': round(to, 2), 'ics': ics})

# 创建DataFrame
df_data = pd.DataFrame(data)

# 打印数据框
print(df_data)

# 保存数据框为Excel文件
df_data.to_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/J_ICS_Values.xlsx', index=False)

# 绘制折线图

# 创建新的图表
plt.figure(figsize=(10, 6))

plt.plot(df_data['Table Number'], df_data['ics'], marker='o', linestyle='-', label='ICS Value')
plt.xlabel('Year')
plt.ylabel('ICS Value')
plt.title('ICS Values for 2007 to 2020')
plt.ylim(0, 0.5)
plt.xticks(range(2, 16), range(2007, 2021))

# 设置刻度标签的旋转角度为45度
plt.xticks(range(2, 16), range(2007, 2021), rotation=45)

plt.legend()

# 提高输出图像的分辨率
plt.savefig('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/J_ICS_Values_Plot.png', dpi=300)

plt.show()


# In[24]:


# 读取中国的ICS值数据
df_china = pd.read_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/ICS_Values.xlsx')

# 读取日本的ICS值数据
df_japan = pd.read_excel('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/J_ICS_Values.xlsx')

# 创建新的图表
plt.figure(figsize=(10, 6))

# 绘制中国的ICS值折线图
plt.plot(df_china['Table Number'], df_china['ics'], marker='o', linestyle='-', label='China')

# 绘制日本的ICS值折线图
plt.plot(df_japan['Table Number'], df_japan['ics'], marker='o', linestyle='-', label='Japan')

# 添加坐标轴标签和图表标题
plt.xlabel('Year')
plt.ylabel('ICS Value')
plt.title('Comparison of ICS Values for China and Japan (2007 to 2020)')

# 设置坐标轴范围和刻度
plt.ylim(0, 0.5)
plt.xticks(range(2, 16), range(2007, 2021))

# 设置刻度标签的旋转角度为45度
plt.xticks(range(2, 16), range(2007, 2021), rotation=45)

# 添加图例
plt.legend()

# 提高输出图像的分辨率
plt.savefig('D:/HBU/1/1_2/01 国民经济核算与投入产出分析/JC_ICS_Values_Plot.png', dpi=300)

# 显示图表
plt.show()


# In[ ]:




