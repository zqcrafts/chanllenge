import numpy as np
from bbox import bbox_overlaps

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
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]

# def py_weighted_nms(det, thresh_lo, thresh_hi):
#     order = det[:, 4].ravel().argsort()[::-1]
#     det = det[order, :]
#     if det.shape[0] == 0:
#         dets = np.array([[10, 10, 20, 20, 0.002]])
#         det = np.empty(shape=[0, 5])
#     while det.shape[0] > 0:
#         # IOU
#         area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
#         xx1 = np.maximum(det[0, 0], det[:, 0])
#         yy1 = np.maximum(det[0, 1], det[:, 1])
#         xx2 = np.minimum(det[0, 2], det[:, 2])
#         yy2 = np.minimum(det[0, 3], det[:, 3])
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         o = inter / (area[0] + area[:] - inter)

#         # nms
#         merge_index = np.where(o >= thresh_lo)[0]
#         det_accu = det[merge_index, :]
#         det = np.delete(det, merge_index, 0)
#         if merge_index.shape[0] <= 1:
#             if det.shape[0] == 0:
#                 try:
#                     dets = np.row_stack((dets, det_accu))
#                 except:
#                     dets = det_accu
#             continue
#         det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
#         max_score = np.max(det_accu[:, 4])
#         det_accu_sum = np.zeros((1, 5))
#         det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
#                                       axis=0) / np.sum(det_accu[:, -1:])
#         det_accu_sum[:, 4] = max_score
#         try:
#             dets = np.row_stack((dets, det_accu_sum))
#         except:
#             dets = det_accu_sum
#     # dets = dets[0:750, :]
#     return dets

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
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

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
        ovr = inter / (areas[i] + areas[order] - inter)
        # ovr = bbox_overlaps(dets[i].astype(np.float).reshape(-1, 5), dets[order].astype(np.float).reshape(-1, 5)).reshape(-1, )

        inds = np.where(ovr <= thresh_lo)[0]
        inds_keep = np.where(ovr > thresh_hi)[0]
        if len(inds_keep) == 0:
            break

        order_keep = order[inds_keep]

        tmp=np.sum(scores[order_keep])
        x1_avg = np.sum(scores[order_keep] * x1[order_keep]) / tmp
        y1_avg = np.sum(scores[order_keep] * y1[order_keep]) / tmp
        x2_avg = np.sum(scores[order_keep] * x2[order_keep]) / tmp
        y2_avg = np.sum(scores[order_keep] * y2[order_keep]) / tmp

        keep.append([x1_avg, y1_avg, x2_avg, y2_avg, scores[i]])
        order = order[inds]
    return np.array(keep)