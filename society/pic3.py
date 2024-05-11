import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 提供的数据
cities = [
    "重庆", "成都", "自贡", "泸州", "德阳",
    "绵阳", "遂宁", "内江", "乐山", "南充",
    "眉山", "宜宾", "广安", "达州", "雅安", "资阳"
]

years = ["2019年", "2020年", "2021年"]

data = [
    [171200, 235600, 178223],
    [123762, 127905, 134895],
    [18514, 18399, 13426],
    [24750, 24898, 24672],
    [18101, 18051, 18820],
    [27265, 27594, 28633],
    [14823, 15922, 14670],
    [18519, 18616, 19012],
    [18874, 18634, 18900],
    [32949, 35108, 36553],
    [12989, 13523, 20413],
    [26470, 27543, 27479],
    [20626, 16577, 20960],
    [34773, 25027, 38991],
    [11262, 11753, 12581],
    [12992, 13385, 14186]
]

# 转换数据格式
X, Y = np.meshgrid(range(len(cities)), range(len(years)))
Z = np.array(data).T

# 创建3D曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面，设置颜色为亮色主题
surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap('viridis', 256))

# 设置轴标签
ax.set_xticks(np.arange(len(cities)) + 0.5)  # 设置刻度的位置
ax.set_xticklabels(cities, rotation=45)  # 设置刻度标签并旋转45度以防止重叠
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years)

ax.set_xlabel(' ')
ax.set_ylabel('年份')
ax.set_zlabel('医院床位数（张）')

# 设置x轴范围
ax.set_xlim(0, len(cities))

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 显示图表
plt.show()
