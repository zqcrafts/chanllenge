import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mutils
from pycocotools.coco import COCO
import cv2
import matplotlib.patches as patches
from matplotlib import colorbar as cbar

# anno_path = '/gdata1/zhuqi/DarkFace_MF_5500/annotations/val_annotations.json'
# img_path = '/gdata1/zhuqi/DarkFace_MF_5500/val/'

anno_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/annotations/val_annotations.json'
img_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/val/'

pred_path = '/gdata2/zhuqi/darkface/tools/merge_annotation/5500_msrcr_zhuqi_ensemble_val.pkl'

num_class = 1

pred_data = mmcv.load(pred_path)
anno_data = mmcv.load(anno_path)

min_det_score = 0.5

def plot_anno_pred(anno_img, img, pred, save_path=None):
    #     plot_anno_pred(anno, img, pred, save_path=save_path)
    min_det_score = 0.5  # 最低阈值
    pred = np.array(pred)
    boxes = pred[:,:,:3].astype(np.int)  # 预测到的box
    scores = pred[:, 4]  # 预测到的分数
    valid_idx = (scores >= min_det_score)  # 判断有效框
    print(valid_idx)
    print(boxes)
    boxes = boxes[valid_idx]; #scores = scores[valid_idx]  # 获得有效框的box和分数
    fig = plt.figure(figsize=(30, 15))  # 画图
    fig.add_subplot(121)  # 1，2，1 一行分成两列，第一个子图
    #plt.imshow(anno_img[:,:,::-1])  # 接收标注图像  ::-1 倒序输出
    fig.add_subplot(122)  # 第二个子图
    ax = plt.gca()  # 挪动坐标轴
    plt.imshow(img[:,:,::-1])  # 接收原图像
    # cmap=plt.cm.Wistia
    cmap = plt.cm.spring  # 获取不同的颜色color map
    normal = plt.Normalize(min_det_score, max(scores))  # 归一化[0,1]，最小值为最低阈值，最大值为score最大值
    colors = cmap(scores)  # 不同分数对应不同颜色
    print(zip(boxes, colors))
    print("#################")
    for box, c in zip(boxes, colors):  # 预测值化为了四个值+color
        rect=patches.Rectangle(box[:2],*box[2:],linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
    #  cax, _ = cbar.make_axes(ax)
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
    pad_fraction = 0.5; aspect = 20
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb2 = cbar.ColorbarBase(cax, cmap=cmap, norm=normal)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)

def draw_bbox(img, bbox_score_data, index, show_score=False):
    global min_det_score

    test = list(map(lambda x: int(x), bbox_score_data[:4])) # map() 根据提供的函数对指定序列做映射,返回函数值列表
    scores = bbox_score_data[4]
    index %= 5

    U = 255 #30 + 255*(scores-0.5)*2 if scores < 0.94 else 255
    V = 0 #30 + 255*(scores-0.5)*2 if scores < 0.94 else 255
    Y = 30 + 255*(scores-0.5)*2 if scores < 0.94 else 255

    R = Y +1.403*(V-128)
    G = Y - 0.343*(U-128)-0.714*(V-128)
    B = Y + 1.77*(U-128)

    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), [B,G,R], 2)

    # if show_score:
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     text = " "
    #     if show_score:
    #         text += str(bbox_score_data[4])[:4]
    #     rec_width = len(text) * 6
    #     img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
    #     # cv2.rectangle(image, start_point, end_point, color, thickness)
    #     # image:它是要在其上绘制矩形的图像。
    #     # start_point：它是矩形的起始坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
    #     # end_point：它是矩形的结束坐标。坐标表示为两个值的元组，即(X坐标值ÿ坐标值)。
    #     # color:它是要绘制的矩形的边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
    #     # thickness:它是矩形边框线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。
    #     cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

def parse_img(img, anno, gt_list):

    bbox_data = anno[0]

    for bbox in bbox_data:#, mask_data[cid]):
        print(bbox[4])
        if bbox[4] > 0.3:
            img = draw_bbox(img, bbox, 1, show_score=True)

    for gt in gt_list:
        x_min = int(gt[0])
        y_min = int(gt[1])
        x_max = int(gt[2]+gt[0])
        y_max = int(gt[3]+gt[1])

        img = cv2.rectangle(img, (x_min,y_min), (x_max, y_max), [0,0,255], 1)

    return img


for index, (anno, pred) in enumerate(zip(anno_data['images'], pred_data)): # zip:将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
   
    gt_list = []
    for gt in anno_data['annotations']:
        if gt["image_id"] == anno["id"]:
            gt_list.append(gt["bbox"])
    print(gt_list)

    img_name = anno['file_name'] 
    
    #print(img_name)
    #print(anno)  # annotationa中的image参数,字典类型{'file_name': '6.png', 'height': 720, 'width': 1080, 'id': 1}
    #print(pred)  # 预测到的n个五个值组成二维列表，[[四个坐标值,score],[..],[..],...,[..]]
   
    img = cv2.imread(img_path + img_name) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = parse_img(img, pred, gt_list)

    save_path = '/gdata2/zhuqi/darkface/tools/merge_annotation/5500_msrcr_zhuqi_ensemble_val/%s' % img_name
    plt.imsave(save_path, img)
    if index % 100 == 0:
        print("Saving %d/%d" % (index, len(pred_data)))