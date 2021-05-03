import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from ...bbox.iou_calculators.iou2d_calculator import bbox_overlaps

@BBOX_ASSIGNERS.register_module()
class ATSSCostAssignerCenterAllTopk(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha=0.8,  # NOTE location hyperparameter alpha
                 beta=0.2,  # NOTE cls hyperparameter beta
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               cls_scores,
               bbox_preds,  # NOTE prediction instead of pre-defined
               bbox_coder,  # NOTE bbox_coder
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        num_gt, num_bboxes, num_bbox_preds = gt_bboxes.size(0), bboxes.size(0), bbox_preds.size(0)
        assert num_bboxes == num_bbox_preds

        # assign 0 by default
        assigned_gt_inds = cls_scores.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = cls_scores.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = cls_scores.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1
            assert True==False


        bbox_repeat = bboxes.unsqueeze(1).repeat(1, num_gt, 1).view(-1, 4)
        bbox_preds_repeat = bbox_preds.unsqueeze(1).repeat(1, num_gt, 1)
        gt_bboxes_repeat = gt_bboxes.unsqueeze(0).repeat(num_bbox_preds, 1, 1)

        bbox_preds_decode = bbox_coder.decode(bbox_repeat, bbox_preds_repeat.view(-1,4)).view(num_bbox_preds, num_gt, 4)
        gt_bboxes_decode = gt_bboxes_repeat
        bbox_preds_encode = bbox_preds_repeat
        gt_bboxes_encode = bbox_coder.encode(bbox_repeat, gt_bboxes_repeat.view(-1,4)).view(num_bbox_preds, num_gt, 4)

        bbox_preds_x = (bbox_preds_decode[:, :, 0] + bbox_preds_decode[:, :, 2]) * 0.5
        bbox_preds_y = (bbox_preds_decode[:, :, 1] + bbox_preds_decode[:, :, 3]) * 0.5
        bbox_preds_center = torch.stack([bbox_preds_x, bbox_preds_y], dim=-1)

        bboxes_repeat = bboxes.unsqueeze(1).repeat(1, num_gt, 1)
        bboxes_x = (bboxes_repeat[:, :, 0] + bboxes_repeat[:, :, 2]) * 0.5
        bboxes_y = (bboxes_repeat[:, :, 1] + bboxes_repeat[:, :, 3]) * 0.5
        bboxes_w = (bboxes_repeat[:, :, 2] - bboxes_repeat[:, :, 0])
        bboxes_h = (bboxes_repeat[:, :, 3] - bboxes_repeat[:, :, 1])
        bboxes_center = torch.stack([bboxes_x, bboxes_y], dim=-1)
        bboxes_wh = torch.stack([bboxes_w, bboxes_h], dim=-1)

        gt_center_x = (gt_bboxes_decode[:, :, 0] + gt_bboxes_decode[:, :, 2]) * 0.5
        gt_center_y = (gt_bboxes_decode[:, :, 1] + gt_bboxes_decode[:, :, 3]) * 0.5
        gt_center = torch.stack([gt_center_x, gt_center_y], dim=-1)
        
        # cls_cost只对每个gt取对应的gt_label进行计算
        cls_scores_repeat = cls_scores[:, gt_labels].sigmoid()

        # giou loss
        cost_bbox_num = bbox_overlaps(bbox_preds_decode, gt_bboxes_decode, mode='giou', is_aligned=True, eps=1e-7)
        cost_bbox_num = 1 - cost_bbox_num

        # l1 loss
        cost_bbox2_num = torch.abs(bbox_preds_encode - gt_bboxes_encode) * 0.5
        cost_bbox2_num = cost_bbox2_num.mean(dim=2)

        # focal loss
        gamma = 2.0
        cost_cls_num = 0.5 * ((1 - cls_scores_repeat) ** gamma) * (-(cls_scores_repeat + 1e-8).log())
        
        # print('cost_bbox_num :', cost_bbox_num)
        # print('cost_bbox2_num :', cost_bbox2_num)
        # print('cost_cls_num :', cost_cls_num)

        assert cost_bbox2_num.shape == cost_cls_num.shape
        assert torch.all(cost_bbox_num >= 0)
        assert torch.all(cost_bbox2_num >= 0)
        assert torch.all(cost_cls_num >= 0)
        overlaps = cost_bbox_num + cost_bbox2_num + cost_cls_num
        
        anchor_distance_cost = torch.abs(bbox_preds_center - bboxes_center) * 0.5
        # NOTE:对anchor_distance_cost进行归一化（按bbox的wh大小分别对xy归一化）
        anchor_distance_cost = anchor_distance_cost / (bboxes_wh / 8)
        # print('anchor_distance_cost :', anchor_distance_cost)
        # 计算L2距离
        anchor_distance_cost = pow(anchor_distance_cost[:, :, 0], 2) + pow(anchor_distance_cost[:, :, 1], 2)
        anchor_distance_cost = pow(anchor_distance_cost, 0.5)

        anchor_distance_ratio = (0.5 * anchor_distance_cost + torch.full_like(anchor_distance_cost, 1))
        assert torch.all(anchor_distance_ratio >= 0)
        overlaps = overlaps * anchor_distance_ratio
        # print('anchor_distance_ratio :', anchor_distance_ratio)

        # NOTE: 确保predicted和anchor的shape，并在索引时一一对应，因为overlap是根据predicted排序的，而assign的时候是根据overlap的顺序assign给anchor
        assert overlaps.shape[0] == assigned_gt_inds.shape[0]

        overlaps_thr_per_gt = overlaps.topk(self.topk, dim=0, largest=False)[0][self.topk - 1]
        is_pos = overlaps <= overlaps_thr_per_gt[None, :]

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        # limit the positive sample's center in gt
        ep_bboxes_cx = bboxes_cx.view(-1, 1).expand(
            num_bboxes, num_gt).contiguous()
        ep_bboxes_cy = bboxes_cy.view(-1, 1).expand(
            num_bboxes, num_gt).contiguous()

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        overlaps_inf = torch.full_like(overlaps,
                                       INF).contiguous().view(-1)
        index = is_pos.view(-1)
        overlaps_inf[index] = overlaps.contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(-1, num_gt)

        # NOTE: 第一次assign，用于计算第一次的rank
        max_overlaps, argmax_overlaps = overlaps_inf.min(dim=1)
        assigned_gt_inds[
            max_overlaps != INF] = argmax_overlaps[max_overlaps != INF] + 1
            
        # get the cost ranks of bboxes
        ranked_values, indices = overlaps_inf.topk(k=num_bboxes, dim=0, largest=False)
        assert torch.all(ranked_values >= 0)
        ranked_values = ranked_values % INF
        
        ranks = torch.full_like(overlaps_inf ,-1).contiguous()
        for i in range(num_gt):
            ranks[indices[:, i], i] = torch.arange(num_bboxes).type_as(overlaps_inf)
        assert torch.all(ranks != -1)

        sort_dim0 = torch.where(assigned_gt_inds > 0)[0]

        conflict_num = (overlaps_inf != INF).sum(dim=1)
        conflict_sort_dim0 = (conflict_num > 1)  # 有冲突的anchor的dim0索引
        sort_dim1 = torch.full_like(assigned_gt_inds, -1)
        sort_dim1[sort_dim0] = argmax_overlaps[sort_dim0]
        
        # NOTE: 如果一个anchor分配给多个gt，取rank最高的那个gt
        if len(conflict_sort_dim0) > 0 :
            ranks_inf = torch.full_like(ranks, INF).contiguous().view(-1)
            overlaps_conflict = torch.full_like(overlaps_inf, -INF)

            overlaps_conflict[conflict_sort_dim0] = overlaps_inf[conflict_sort_dim0]
            dim0, dim1 = torch.where(overlaps_conflict > 0)
            index = dim0 * num_gt + dim1
            ranks_inf[index] = ranks.view(-1)[index]
            ranks_inf = ranks_inf.view(num_bboxes, num_gt)

            _, conflict_valid_dim1 = torch.min(ranks_inf, dim=1)
            sort_dim1[conflict_sort_dim0] = conflict_valid_dim1[conflict_sort_dim0]
        
        sort_dim1 = sort_dim1[sort_dim0]
        assert torch.all(sort_dim1 != -1)
        assert sort_dim0.shape == sort_dim1.shape

        # NOTE: 第二次assign，根据rank排序后得到
        assigned_gt_inds[sort_dim0] = sort_dim1 + 1
        max_overlaps[sort_dim0] = overlaps_inf[sort_dim0, sort_dim1]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
