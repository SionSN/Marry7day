import numpy as np
import pandas as pd
N=6
T=4
data1=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'人口密度（人 平方公里）').iloc[:,1:4]
data2=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'建成区绿化覆盖率').iloc[:,1:4]
data3=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'第一产业增加值增长率').iloc[:,1:4]
data4=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'第二产业增加值增长率').iloc[:,1:4]
data5=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'第三产业增加值增长率').iloc[:,1:4]
data6=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'城市污水处理率').iloc[:,1:4]
data7=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'水资源总量').iloc[:,1:4]
data8=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'人均GDP指数').iloc[:,1:4]
data9=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'人均公园绿地面积').iloc[:,1:4]
data10=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'从业人员数').iloc[:,1:4]
data11=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'GDP增长率')
data12=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'专利授权量').iloc[:,1:4]
data13=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'出口总额').iloc[:,1:4]
data14=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'粮食产量').iloc[:,1:4]
data15=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'进口总额').iloc[:,1:4]
data16=pd.read_excel("D:\\pythonProject1\\data\\alldata.xlsx",'大中型工业企业数').iloc[:,1:4]

data_list= [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11.iloc[:,1:4], data12, data13, data14, data15, data16]
# 第一步：构造数据矩阵
# 提取每个 data 中的第一列，组合形成一个矩阵
data_matrix_2019 = np.column_stack([data.iloc[:, 0] for data in data_list])
data_matrix_2020 = np.column_stack([data.iloc[:, 1] for data in data_list])
data_matrix_2021 = np.column_stack([data.iloc[:, 2] for data in data_list])
# data_matrix_2022 = np.column_stack([data.iloc[:, 3] for data in data_list])
matrix_list= [data_matrix_2019, data_matrix_2020, data_matrix_2021]
# print(data_matrix_2019)
values = np.zeros((3,16))
for i, data_matrix in enumerate(matrix_list):
    # 将字符串类型的元素转换为浮点数
    data_matrix = data_matrix.astype(float)
    # 对 data_matrix 进行按列计算 Zij
    z_matrix = data_matrix / np.sum(data_matrix**2, axis=0)
    # 将概率值限制在一个很小的正数范围内，避免零或负数
    p_matrix = z_matrix / np.sum(z_matrix, axis=0)
    # 计算信息熵值 ej
    entropy_values = -1/np.log(16) * np.sum(p_matrix * np.log(p_matrix), axis=0)
    # 计算信息效用值 dj
    utility_values = 1 - entropy_values
    # 计算评价指标权重 Wj
    weight_values = utility_values / np.sum(utility_values)
    # 假设已经有了 Pij 和 Wj
    # Pij 是概率矩阵，Wj 是评价指标权重

    # 计算 DEDCIi
    dedci_values = np.sum(weight_values * p_matrix, axis=1)
    values[i,:]=dedci_values

print("weight_values:\n", weight_values)
print("values:\n", values)

import numpy as np
#
# def ahp(matrix):
#     # 步骤1：对矩阵进行标准化
#     normalized_matrix = matrix / matrix.sum(axis=0)
#
#     # 步骤2：计算加权矩阵
#     weighted_matrix = normalized_matrix / normalized_matrix.sum(axis=0)
#     # 步骤3：计算特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eig(weighted_matrix)
#
#     # 步骤4：计算主特征向量
#     principal_eigenvector = eigenvectors[:, np.argmax(np.real(eigenvalues))]
#
#     # 步骤5：对主特征向量进行标准化
#     weights = principal_eigenvector / np.sum(principal_eigenvector)
#
#     return weights
#
# # 示例用法：
# # 用您的两两比较矩阵替换以下矩阵
# pairwise_matrix = np.array([
#     # data_matrix_2019,
#     # data_matrix_2020,
#     data_matrix_2021,
#
# ])
# # pairwise_matrix = np.mean(pairwise_matrix, axis=1)
# # 调用ahp函数并传入两两比较矩阵
# weights = ahp(pairwise_matrix)
#
# # 打印计算得到的权重
# print("计算得到的权重:", weights)
#

# 假设values矩阵已经被创建
# 假设地区和年份的列表已经存在
regions = ["重庆", "成都", "自贡", "泸州", "德阳", "绵阳", "遂宁", "内江", "乐山", "南充", "眉山", "宜宾", "广安", "达州", "雅安", "资阳"]
years_2019 = values[0]
years_2020 = values[1]
years_2021 = values[2]
# years_2022 = values[3]


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码

x = np.arange(len(regions))  # 标签位置
width = 0.2  # 柱状图的宽度，可以根据自己的需求和审美来改

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width * 2, years_2019, width, label='2019年', color='#b4edab')
rects2 = ax.bar(x - width + 0.01, years_2020, width, label='2020年', color='#cee55d')
rects3 = ax.bar(x + 0.02, years_2021, width, label='2021年', color='#6adc88')
# rects4 = ax.bar(x + width + 0.03, years_2022, width, label='2022年',color='#f8cb7f')


# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('综合评价指数', fontsize=16)
ax.set_xlabel('城市', fontsize=16)
ax.set_title('2019年-2021年成渝地区双城区域发展水平综合评价指数',fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(regions,fontsize=10)
ax.legend(loc='upper right')
fig.tight_layout()
# plt.scatter(regions, values[2], label='2021年', color='#cee55d')  # 绘制每个数据点
plt.plot(regions, values[2], label='2021年', color='#81b3a9')  # 绘制折线
# 标注数据值
for i, txt in enumerate(values[2]):
    plt.text(regions[i], txt, '{:.4f}'.format(txt), ha='center', va='bottom', fontsize=10, color='#333333')

# # 添加图例
plt.legend()
# plt.savefig('2019年-2021年成渝地区双城区域发展水平综合评价指数.png')#保存图片
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 用于正常显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 各个属性值
# feature = regions
# value = values
#
# # 设置每个数据点的显示位置，在雷达图上用角度表示
# angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
# angles = np.concatenate((angles, [angles[0]]))
# feature = np.concatenate((feature, [feature[0]]))
#
#
# # 绘图
# fig = plt.figure(facecolor='white')
# index0 = 0
# subject_label = ['2019年', '2020年', '2021年',]
# for values in [value[0, :], value[1, :], value[2, :]]:
#     # 拼接数据首尾，使图形中线条封闭
#     values = np.concatenate((values, [values[0]]))
#     # 设置为极坐标格式
#     ax = fig.add_subplot(111, polar=True)
#     # 绘制折线图
#     ax.plot(angles, values, 'o-', linewidth=1, label=subject_label[index0])
#     # 填充颜色
#     ax.fill(angles, values, alpha=0.25)
#
#     # 设置图标上的角度划分刻度，为每个数据点处添加标签
#     ax.set_thetagrids(angles[:-1] * 180 / np.pi, feature[:-1])
#
#     # 设置雷达图的范围
#     ax.set_ylim(0, 0.2)
#     index0 = index0 + 1
#
# # 添加标题
# plt.title('成渝双城地区发展水平星图')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1, frameon=False)
# # 添加网格线
# ax.grid(True)
# # plt.savefig('雷达图.png')#保存图片
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 假设有16个横坐标
# x_values = regions
#
#
# # 假设有四组数据，每组数据有16个纵坐标值
# data_1 = values[0]
# data_2 = values[1]
# data_3 = values[2]
# # data_4 = data11.iloc[:,4]
#
# # 设置画布大小
# plt.figure(figsize=(10, 6))
# # 创建折线图
# # plt.plot(x_values, data_1, label='2019年')
# # plt.plot(x_values, data_2, label='2020年')
# plt.plot(x_values, data_3, label='2021年',color='#53abd8')
# # plt.plot(x_values, data_4, label='2022年')
#
# # 添加标签和标题
# plt.xlabel('地区')
# plt.ylabel('GDP增长率')
# plt.title('2021年成渝地区双城区域发展水平综合评价指数')
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()

