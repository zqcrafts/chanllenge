
import pandas as pd
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['figure.figsize'] = (10.0, 10.0)


# 读取数据
ann_json = '/gdata1/zhuqi/Darkface_coco_MF/annotations/train_annotations.json'
with open(ann_json) as f:
    ann=json.load(f)

#################################################################################################
#创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
counts_label=dict([(i['name'],0) for i in ann['categories']])
for i in ann['annotations']:
    counts_label[category_dic[i['category_id']]]+=1

# 标注长宽高比例
box_w = []
box_h = []
box_wh = []
categorys_wh = [[] for j in range(10)]
for a in ann['annotations']:
    print(a['image_id'])
    if a['category_id'] != 0 and a['bbox'][2] != 0 and a['bbox'][3] != 0:
        # print(a['bbox'][0])
        # print(a['bbox'][1])
        # print(a['bbox'][2])

        # if(a['bbox'][3] == 0):
        #     print(a['image_id'])
        #     print(a['bbox'])
        #     print(a['id'])
        box_w.append(round(a['bbox'][2],2)) # W
        box_h.append(round(a['bbox'][3],2)) # H
        wh = round(a['bbox'][3]/a['bbox'][2],1) # H/W
        # if wh <1 :
        #     wh = round(a['bbox'][3]/a['bbox'][2],1) # round:将数值四舍五入到1个小数位
        box_wh.append(wh)

        categorys_wh[a['category_id']-1].append(wh)


# 所有标签的长宽高比例
box_wh_unique = list(set(box_wh))
box_wh_count=[box_wh.count(i) for i in box_wh_unique]

# 绘图

print(box_wh_count)
print(box_wh_unique)

plt.bar(box_wh_unique,box_wh_count,width=0.05,color="green") # width：柱的宽度
plt.xticks(box_wh_unique)
plt.yticks(box_wh_count) # y轴坐标刻度
plt.xlim(0.2,4)  # 刻度范围1~4
#plt.legend(fontsize=1)  # 刻度字体大小
plt.tick_params(labelsize=5) #刻度值字体大小设置（x轴和y轴同时设置）
plt.xlabel("ratio")
plt.ylabel("num")
plt.title("anchor_histgram")

plt.savefig('/gdata2/zhuqi/darkface/tools/caculator/anchor_ratio_histgram.jpg')