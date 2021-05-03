import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


# def multiclass_nms(multi_bboxes,
#                    multi_scores,
#                    score_thr,
#                    nms_cfg,
#                    max_num=-1,
#                    score_factors=None):
#     """NMS for multi-class bboxes.

#     Args:
#         multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
#         multi_scores (Tensor): shape (n, #class), where the last column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms_thr (float): NMS IoU threshold
#         max_num (int): if there are more than max_num bboxes after NMS,
#             only top max_num will be kept.
#         score_factors (Tensor): The factors multiplied to scores before
#             applying NMS

#     Returns:
#         tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
#             are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 4:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
#     else:
#         bboxes = multi_bboxes[:, None].expand(
#             multi_scores.size(0), num_classes, 4)

#     scores = multi_scores[:, :-1]
#     if score_factors is not None:
#         scores = scores * score_factors[:, None]

#     labels = torch.arange(num_classes, dtype=torch.long)
#     labels = labels.view(1, -1).expand_as(scores)

#     bboxes = bboxes.reshape(-1, 4)
#     scores = scores.reshape(-1)
#     labels = labels.reshape(-1)

#     # remove low scoring boxes
#     valid_mask = scores > score_thr
#     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
#     if inds.numel() == 0:
#         if torch.onnx.is_in_onnx_export():
#             raise RuntimeError('[ONNX Error] Can not record NMS '
#                                'as it has not been executed this time')
#         return bboxes, labels

#     # TODO: add size check before feed into batched_nms
#     dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

#     if max_num > 0:
#         dets = dets[:max_num]
#         keep = keep[:max_num]

#     return dets, labels[keep]

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):

    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.get('type', 'nms')
    # NOTE for NO NMS only
    with_nms = nms_cfg_.get('with_nms', True)
    if 'with_nms' in nms_cfg_:
        nms_cfg_.pop('with_nms')
    if with_nms == False:
        return _multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg_,
                               max_num, score_factors, with_nms=with_nms)
    if nms_type.startswith('lb_'):
        nms_cfg_['type'] = nms_type[3:]
        return _lb_multiclass_nms(multi_bboxes, multi_scores, score_thr,
                                  nms_cfg_, max_num, score_factors)
    else:
        return _multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg_,
                               max_num, score_factors)


def _multiclass_nms(multi_bboxes,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None,
                    with_nms=True):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # NOTE for NO NMS only
    if with_nms == False:
        dets = torch.cat([bboxes, scores.view(-1, 1)], dim=-1)
        return dets, labels
    else:
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, labels[keep]

def _lb_multiclass_nms(multi_bboxes,
                       multi_scores,
                       score_thr,
                       nms_cfg,
                       max_num=-1,
                       score_factors=None,
                       others=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    return_others = True
    if others is None:
        return_others = False
        others = multi_bboxes.new_zeros(multi_bboxes.shape)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)

    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    others = others[valid_mask.view(-1,)]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        others = multi_bboxes.new_zeros((0, 4))

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_others == False:
            return bboxes, labels
        else:
            return bboxes, labels, others
    inds = scores.argsort(descending=True)
    bboxes = bboxes[inds]
    scores = scores[inds]
    labels = labels[inds]
    others = others[inds]

    batch_bboxes = torch.empty((0, 4),
                               dtype=bboxes.dtype,
                               device=bboxes.device)
    batch_scores = torch.empty((0, ), dtype=scores.dtype, device=scores.device)
    batch_labels = torch.empty((0, ), dtype=labels.dtype, device=labels.device)
    batch_others = torch.empty((0, 4),
                               dtype=bboxes.dtype,
                               device=bboxes.device)
    while bboxes.shape[0] > 0:
        num = min(100000, bboxes.shape[0])
        batch_bboxes = torch.cat([batch_bboxes, bboxes[:num]])
        batch_scores = torch.cat([batch_scores, scores[:num]])
        batch_labels = torch.cat([batch_labels, labels[:num]])
        batch_others = torch.cat([batch_others, others[:num]])
        bboxes = bboxes[num:]
        scores = scores[num:]
        labels = labels[num:]
        others = others[num:]

        _, keep = batched_nms(batch_bboxes, batch_scores, batch_labels,
                              nms_cfg)
        batch_bboxes = batch_bboxes[keep]
        batch_scores = batch_scores[keep]
        batch_labels = batch_labels[keep]
        batch_others = batch_others[keep]

    dets = torch.cat([batch_bboxes, batch_scores[:, None]], dim=-1)
    labels = batch_labels
    others = torch.cat([batch_others, batch_scores[:, None]], dim=-1)

    if max_num > 0:
        dets = dets[:max_num]
        labels = labels[:max_num]
        others = others[:max_num]
    if return_others == False:
        return dets, labels
    else:
        return dets, labels, others

def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
