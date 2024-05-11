import numpy as np
import math

# 输入数据
cities = ['重庆', '成都', '自贡', '泸州', '德阳', '绵阳', '遂宁', '内江', '乐山', '南充', '眉山', '宜宾', '广安', '达州', '雅安', '资阳']
num_cities = len(cities)
data =dict()

# 已知的权重数据
w1 = np.array([0.91791, 0.55599, 0.07870, 0.11524, 0.07677, 0.14989, 0.05082, 0.04994, 0.19741, 0.14835, 0.08020, 0.18741, 0.07398, 0.09815, 0.12549, 0.0])
w2 = np.array([0.25128, 0.21052, 0.03707, 0.04411, 0.03490, 0.04830, 0.02826, 0.02681, 0.05507, 0.04541, 0.03851, 0.05679, 0.02984, 0.03286, 0.04384, 0.01643])

fenmu = 0

for i in range(len(cities)):
    fenmu += math.sqrt(w1[i] * w2[i])

for i in range(len(cities)):
    data[cities[i]+"市"] = (math.sqrt(w1[i] * w2[i])+0.001)/fenmu

print(data)

import pandas as pd
from pyecharts.globals import GeoType
from pyecharts.charts import Geo, Map
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
import json
import numpy as np
import os
# json数据文件目录文件夹
file_path = r'全国地图_JSON'
os.chdir(file_path)
# json地图数据的构建
target_data = """{
        "type": "FeatureCollection",
        "features": []}"""
target_data = json.loads(target_data)
# 将云川渝的数据节点传入target_data，并加入area节点
list_region = ['四川省.json', '重庆市.json', '中华人民共和国.json']
name_list = []
for lr in list_region:
    with open(lr, 'r', encoding='utf-8') as f:
        temp = json.load(f)
        temp = temp['features']

        for tp in temp:
            name = tp['properties']['name']
            if name in data:
                # 将area加入tp节点并传入target_data
                tp['properties']['area'] = '川渝'
                target_data['features'].append(tp)
                name_list.append((name, data[name]))

with open('test.json','w',encoding='utf-8') as f:
    f.write(json.dumps(target_data, ensure_ascii=False))


region_name = '2022年成渝地区双城经济圈各城市文化建设水平指数'
c = (
    Map()
    # 注册地图,test_01是注册的地图信息，尽量不要与pyecharts中的地图有任何重名
    .add_js_funcs("echarts.registerMap('test_O1', {});".format(target_data))
    .add(region_name, name_list, "test_O1")
    .set_global_opts(
        title_opts=opts.TitleOpts(title=region_name), visualmap_opts=opts.VisualMapOpts(max_=0.5)
    )
    .render("2022年成渝地区双城经济圈各城市文化建设水平指数.html"))

