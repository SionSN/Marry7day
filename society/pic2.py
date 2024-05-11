import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.globals import ThemeType

# 数据
data = {
    '重庆': [15412, 15956, 30582, 29894, 3209],
    '成都': [25582, 16186, 28324, 21802, 2094.7],
    '自贡': [9484, 12836, 34694, 33570, 317.8387],
    '泸州': [8410, 12186, 31690, 37582, 507.9469],
    '德阳': [11882, 13114, 32693, 32326, 382.2746],
    '绵阳': [12959, 13759, 32089, 30873, 528.5126],
    '遂宁': [7475, 12933, 35840, 32847, 359.31],
    '内江': [8057, 12290, 36909, 33370, 404.9054],
    '乐山': [10584, 12834, 32642, 34703, 348.0004],
    '南充': [8825, 13348, 33385, 32568, 719.2606],
    '眉山': [9600, 11957, 34277, 34095, 341.7565],
    '宜宾': [9621, 12325, 31749, 37020, 551.0433],
    '广安': [6508, 12189, 34110, 36644, 455.5833],
    '达州': [6920, 13781, 38231, 32018, 652.84],
    '雅安': [10649, 12131, 35654, 31762, 152.6056],
    '资阳': [6177, 10195, 37083, 35685, 338.9236]
}

list0 = []
list1 = []
list2 = []
list3 = []

# 计算每个地区的不同学历人口的占比
for city, values in data.items():
    total_population = values[-1]
    list0.append({"value": values[0] / 100000 * total_population, "percent": values[0] / 100000 * 100})
    list1.append({"value": values[1] / 100000 * total_population, "percent": values[1] / 100000 * 100})
    list2.append({"value": values[2] / 100000 * total_population, "percent": values[2] / 100000 * 100})
    list3.append({"value": values[3] / 100000 * total_population, "percent": values[3] / 100000 * 100})

c = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    .add_xaxis(list(data.keys()))
    .add_yaxis("大专及以上", list0, stack="stack1", category_gap="60%", bar_width="80%")  # 调整柱子宽度
    .add_yaxis("高中和中专", list1, stack="stack1", category_gap="60%", bar_width="80%")  # 调整柱子宽度
    .add_yaxis("初中", list2, stack="stack1", category_gap="60%", bar_width="80%")  # 调整柱子宽度
    .add_yaxis("小学", list3, stack="stack1", category_gap="60%", bar_width="80%")  # 调整柱子宽度
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .reversal_axis()
    .set_global_opts(
        yaxis_opts=opts.AxisOpts(name="城市"),  # Y轴标签，表示城市
        xaxis_opts=opts.AxisOpts(name="人数（万人）")  # X轴标签
    )
    .render("受教育程度.html")  # 将图表输出为 HTML 文件
)
