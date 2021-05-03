import numpy as np
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]  # 最高分的序数
        keep.append(i)  # keep保存i值

        xx1 = np.maximum(x1[i], x1[order[1:]])  # 将最高分和其他框进行比较
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # 得到最大交集
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # 求与其他框的iou，并按分数排列

        inds = np.where(iou <= thresh)[0]  # where(iou <= thresh)返回满足条件的索引，返回是一个元组（[]）,所以取第一个[0]
        order = order[inds + 1]  # 把重复框舍弃掉，取不重复的框继续计算

    return dets[keep, :]


def py_weighted_nms(dets, thresh_lo, thresh_hi):
    """
    voting boxes with confidence > thresh_hi
    keep boxes overlap <= thresh_lo
    rule out overlap > thresh_hi
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh_lo: retain overlap <= thresh_lo
    :param thresh_hi: vote overlap > thresh_hi
    :return: indexes to keep
    """

    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按照分数从大到小排列框的序号

    keep = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter)  # 求与其他框的iou，并按分数排列
        # ovr = bbox_overlaps(dets[i].astype(np.float).reshape(-1, 5), dets[order].astype(np.float).reshape(-1, 5)).reshape(-1, )

        inds = np.where(ovr <= thresh_lo)[0]  #取所有iou没超过限制的框的序列
        # where返回的是一个元组，里面包含array类型，如(array([0, 2, 4, 6], dtype=int64),) ，所以取[0]后，才是array([0, 2, 4, 6], dtype=int64)
        inds_keep = np.where(ovr > thresh_hi)[0]  # 取所有满足iou超过限制的框的序列，这是正常nms要舍弃的框，这里用来做加权
        if len(inds_keep) == 0:  # 如果满足iou超过限制的框就是自己，则退出
            break

        order_keep = order[inds_keep]  # 取得用来加权框的序号

        tmp=np.sum(scores[order_keep])  # 取得这些框做加权
        x1_avg = np.sum(scores[order_keep] * x1[order_keep]) / tmp
        y1_avg = np.sum(scores[order_keep] * y1[order_keep]) / tmp
        x2_avg = np.sum(scores[order_keep] * x2[order_keep]) / tmp
        y2_avg = np.sum(scores[order_keep] * y2[order_keep]) / tmp

        keep.append([x1_avg, y1_avg, x2_avg, y2_avg, scores[i]])  # 分数还是取原来的最高分数，这里跟论文好像不太一样
        order = order[inds]  # 把重复框舍弃掉，取不重复的框继续计算
    return np.array(keep)  # 返回二维array列表


def zhuqi_ensemble(dets1, dets2, thr, score_thr): # 起到消除多余框的作用，但也会消除对的框

    dets1 = np.array(dets1)
    dets2 = np.array(dets2)
    # 把模型A中没有和模型B相交iou>thr的框删除
    inds1 = []
    for i in range(len(dets1)):

        areas1 = (dets1[i,2] - dets1[i, 0]) * (dets1[i, 3] - dets1[i, 1])

        for j in range(len(dets2)):
            areas2 = (dets2[j, 2] - dets2[j, 0]) * (dets2[j, 3] - dets2[j, 1])
            xx1 = np.maximum(dets1[i,0], dets2[j,0])
            yy1 = np.maximum(dets1[i, 1], dets2[j, 1])
            xx2 = np.minimum(dets1[i, 2], dets2[j, 2])
            yy2 = np.minimum(dets1[i, 3], dets2[j, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            #print(inter)
            iou = inter / (areas1 + areas2 - inter)
            #print(iou)
            #print(thr)

            if iou > thr and dets2[j, 4] > 0.3 :
                inds1.append(i)
                break
            if iou <= thr and j == len(dets2)-1 :# and dets1[i, 4] > score_thr:
                #inds1.append(i)
                #print(i)
                break

    #把模型B中没有和模型A相交iou>thr的框删除
    inds2 = []
    for i in range(len(dets2)):

        areas1 = (dets2[i, 2] - dets2[i, 0]) * (dets2[i, 3] - dets2[i, 1])

        for j in range(len(dets1)):
            areas2 = (dets1[j, 2] - dets1[j, 0]) * (dets1[j, 3] - dets1[j, 1])
            xx1 = np.maximum(dets2[i, 0], dets1[j, 0])
            yy1 = np.maximum(dets2[i, 1], dets1[j, 1])
            xx2 = np.minimum(dets2[i, 2], dets1[j, 2])
            yy2 = np.minimum(dets2[i, 3], dets1[j, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas1 + areas2 - inter)

            if iou > thr :
                inds2.append(i)
                break
            if iou <= thr and j == len(dets1)-1 and dets2[i, 4] > score_thr:
                inds2.append(i)
                break

    #print(inds1)
    # print(55555555555)
    # print(inds2)
    dets1 = dets1[inds1]
    dets2 = dets2[inds2]

    dets1 = torch.tensor(dets1)
    dets2 = torch.tensor(dets2)
    dets = torch.cat((dets1, dets2), 0)

    #dets = np.array(dets)
    dets1 = np.array(dets1)
    #print(dets.shape)
    #print(dets1.shape)
    #ind = np.where(dets1[:, 4]>0.3)
    #print(dets1[ind])
    return dets1


def choose_size_ensemble(dets,size_thr_l,size_thr_h):
    inds = []
    for i in range(len(dets)):
        areas = (dets[i, 2] - dets[i, 0]) * (dets[i, 3] - dets[i, 1])
        if areas >= size_thr_l and areas < size_thr_h:
            inds.append(i)


    # print(dets)
    # print(inds)
    # print(dets[inds])
    # print(222222222222)
    return dets[inds]

def merge_size_ensemble(det_s,det_m,det_l):
    s = choose_size_ensemble(det_s, 0, 32**2)
    m = choose_size_ensemble(det_m, 32**2, 96*2)
    l = choose_size_ensemble(det_l, 96**2, 250**2 )

    s = torch.tensor(s)  # 合并3个模型的pre
    m = torch.tensor(m)
    l = torch.tensor(l)

    pre_list = torch.cat((s, m, l), 0)
    #pre_list = np.array(pre_list)

    #print(pre_list)
    return pre_list


if __name__ == '__main__':


    parser = argparse.ArgumentParser()  # 参数解析器实例化
    parser.add_argument('--pkl1_path', type=str, default=None)
    parser.add_argument('--pkl2_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    result_list = [] # list[]是一个空的，没有一个元素，进行list[0]就会出现错误！

    #pickle_file = open('/gdata2/zhuqi/work_dirs/tinaface/5500/GN_MF_Diou_Incep_atss_dcn_anchor_cost/val_1100_1500.pkl', 'rb')
    #pickle_file = open('/gdata2/zhuqi/work_dirs/tinaface/5500/GN_MSRCR_Diou_Incep_atss_dcn_anchor_cost/1100_1500_val.pkl', 'rb')
    pkl1_path = args.pkl1_path
    pickle_file = open(pkl1_path, 'rb')
    pre_list1 = pickle.load(pickle_file)  # 导入pkl文件到列表形式[图片数][0][预测个数][xyxys]
    pickle_file.close()

    # print(type(pre_list1[0])) # list
    # print(type(pre_list1[0][0])) # numpy.ndarray
    # print(type(pre_list1[0][0][0])) # numpy.ndarray'
    # print(type(pre_list1[0][0][0][0])) # numpy.float64
    # print(5555555555555)
    #pickle_file = open('/gdata2/zhuqi/work_dirs/tinaface/5500/GN_MF_Diou_Incep_atss_dcn_anchor/val_1100_1700.pkl', 'rb')
    #pickle_file = open('/gdata2/zhuqi/work_dirs/tinaface/5500/GN_MSRCR_Diou_Incep_atss_dcn_anchor/1100_1700_val.pkl', 'rb')
    pkl2_path = args.pkl2_path
    pickle_file = open(pkl2_path, 'rb')
    pre_list2 = pickle.load(pickle_file) 
    pickle_file.close()


    # pre_list1 = torch.tensor(pre_list1)  # 合并两个模型的pre
    # pre_list2 = torch.tensor(pre_list2)
    # pre_list = torch.cat((pre_list1, pre_list2), 2)

    pre = []
    for i in range(len(pre_list2)):
        pre1 = pre_list1[i][0].tolist()
        pre2 = pre_list2[i][0].tolist()
        pre1.extend(pre2)
        pre.append(pre1)

    with tqdm(total=len(pre_list1)) as load_bar:
        for j in range(len(pre_list1)):  # 遍历我们的pre
            #print(j)
            #r = open(root_path + str(j) + '.txt', 'w', encoding='utf-8')


            #pre_nms = zhuqi_ensemble(pre_list1[j][0], pre_list2[j][0], 0.6, 0.9)  # 使用’找不同‘过滤法
            #pre_nms = py_weighted_nms(pre, 0.6, 0.6) 


            #pre_nms = merge_size_ensemble(pre_list1[j][0], pre_list2[j][0], pre_list1[j][0])
            # pre_nms = py_weighted_nms(pre_nms, 0.5, 0.5) 

            pre_nms = py_weighted_nms(pre[j], 0.6, 0.6)  # 使用weighted nms 0.6 0.6 81.4
            #pre_nms = nms(pre[j], 0.5)  # 使用nms


            # for i in range(len(pre_nms)):

            #     x_min = pre_nms[i][0]
            #     y_min = pre_nms[i][1]
            #     x_max = pre_nms[i][2]
            #     y_max = pre_nms[i][3]
            #     score = pre_nms[i][4]

            #     r.writelines(str(('%.6f' % x_min)) + ' ')
            #     r.writelines(str(('%.6f' % y_min)) + ' ')
            #     r.writelines(str(('%.6f' % x_max)) + ' ')
            #     r.writelines(str(('%.6f' % y_max)) + ' ')
            #     r.writelines(str(('%.6f' % score)) + '\n')

            result_list.append([pre_nms])

            # print(len(result_list[0]))
            # print(len(result_list[0][0]))
            # print(len(result_list[0][0][0]))

            load_bar.update(1)  # 进度条走一位

        save_path = args.save_path
        save_pkl_path = save_path
        w = open(save_pkl_path, "wb")
        pickle.dump(result_list, w)