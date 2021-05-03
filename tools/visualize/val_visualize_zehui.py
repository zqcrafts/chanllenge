import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mutils
from pycocotools.coco import COCO
import cv2
import matplotlib.patches as patches
from matplotlib import colorbar as cbar

pred_path = '/gdata2/zhuqi/work_dirs/tinaface/resume60/baseline_test.pkl'
anno_path = '/gdata1/zhuqi/DarkFace_coco_0.666/annotations/val_annotations.json'
img_path = '/gdata1/zhuqi/DarkFace_coco_0.666/val/'
num_class = 1

pred_data = mmcv.load(pred_path)
anno_data = mmcv.load(anno_path)
coco = COCO(anno_path)

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255],\
              [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
ratio = 1

def plot_anno_pred(anno_img, img, pred, save_path=None):
    min_det_score = 0.5
    boxes, scores = pred[:, :4].astype(np.int), pred[:, 4]
    valid_idx = scores >= min_det_score
    boxes = boxes[valid_idx]; scores = scores[valid_idx]
    fig = plt.figure(figsize=(30, 15))
    fig.add_subplot(121)
    plt.imshow(anno_img[:,:,::-1])
    fig.add_subplot(122)
    ax = plt.gca()
    plt.imshow(img[:,:,::-1])
    # cmap=plt.cm.Wistia
    cmap = plt.cm.spring
    normal = plt.Normalize(0.5, max(scores))
    colors = cmap(scores)
    for box, c in zip(boxes, colors):
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
    global color_list
    index %= 5
    #print(bbox_score_data[:4])
    #print(bbox_score_data[:4])
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    #print(test)  # [742]
    #print('8888888888888888')

    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

def parse_img(img, anno):

    bbox_data = anno[0]
    #print("#############")
    #print(bbox_data)  # 这个地方没有读到信息

    for bbox in bbox_data:#, mask_data[cid]):
        print(bbox[4])
        if bbox[4] > 0.3:
            img = draw_bbox(img, bbox, 1, show_score=True)
    return img


for index, (anno, pred) in enumerate(zip(anno_data['images'], pred_data)):

    img_name = anno['file_name'] #+ '.png'这里读到的file_name已经有后缀了
    #print(img_name)
    #print(anno)  # anno只包含annotationa中的image参数
    #print(pred)  # 预测到的五个值[7.42111328e+02, 4.01652344e+02, 7.80128418e+02, 4.42174316e+02,9.34274435e-01],这些值应该是四个坐标

    img = cv2.imread(img_path + img_name) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #print(len(pred)) 
    #print("#########################")


    img = parse_img(img, pred)
    save_path = '/gdata2/zhuqi/work_dirs/tinaface/resume60/pre_visualize/%s' % img_name

    plt.imsave(save_path, img)
    if index % 100 == 0:
        print("Saving %d/%d" % (index, len(pred_data)))