import numpy as np

# 输入数据
cities = ['重庆', '成都', '自贡', '泸州', '德阳', '绵阳', '遂宁', '内江', '乐山', '南充', '眉山', '宜宾', '广安', '达州', '雅安', '资阳']
num_cities = len(cities)

# 构建成渝地区双城经济圈各城市的文化建设数据
culture_data = np.array([
    [41, 23, 7, 8, 7, 10, 6, 6, 12, 10, 7, 11, 7, 8, 9, 4],  # 文化馆数
    [43, 23, 7, 9, 7, 10, 6, 6, 12, 10, 7, 12, 7, 8, 9, 4],  # 图书馆数
    [130, 172, 15, 17, 12, 15, 8, 6, 14, 11, 17, 19, 5, 4, 14, 2]  # 博物馆数
])

# 归一化处理
culture_data_normalized = culture_data / np.sum(culture_data, axis=1, keepdims=True)

# 计算每个城市的文化建设指数
culture_index = np.mean(culture_data_normalized, axis=0)

# 输出结果
for i in range(num_cities):
    print(f"{cities[i]} 文化建设指数: {culture_index[i]:.5f}")
