import numpy as np

# 原始数据
culture_data = {
    'city': ['重庆', '成都', '自贡', '泸州', '德阳', '绵阳', '遂宁', '内江', '乐山', '南充', '眉山', '宜宾', '广安', '达州', '雅安', '资阳'],
    'culture_center': [41, 23, 7, 8, 7, 10, 6, 6, 12, 10, 7, 11, 7, 8, 9, 4],
    'library': [43, 23, 7, 9, 7, 10, 6, 6, 12, 10, 7, 12, 7, 8, 9, 4],
    'museum': [130, 172, 15, 17, 12, 15, 8, 6, 14, 11, 17, 19, 5, 4, 14, 2]
}

# 将原始数据转换成numpy数组
data_matrix = np.array([culture_data['culture_center'], culture_data['library'], culture_data['museum']])

# 数据标准化
norm_data = data_matrix / np.sqrt(np.sum(data_matrix ** 2, axis=1, keepdims=True))

# 计算权重
entropy = -np.sum(norm_data * np.log(norm_data), axis=1)
weight = (1 - entropy) / np.sum(1 - entropy)

# 加权标准化后的数据矩阵
weighted_norm_data = norm_data * weight.reshape(-1, 1)

# 正理想解和负理想解
ideal_best = np.max(weighted_norm_data, axis=1)
ideal_worst = np.min(weighted_norm_data, axis=1)

# 计算各城市到正理想解和负理想解的距离
distance_to_best = np.sqrt(np.sum((weighted_norm_data - ideal_best.reshape(-1, 1)) ** 2, axis=0))
distance_to_worst = np.sqrt(np.sum((weighted_norm_data - ideal_worst.reshape(-1, 1)) ** 2, axis=0))

# 计算综合评价指数
cultural_index = distance_to_worst / (distance_to_best + distance_to_worst)

# 输出各城市的文化建设指数
for city, index in zip(culture_data['city'], cultural_index):
    print(f"{city} 的文化建设指数为: {index:.5f}")
