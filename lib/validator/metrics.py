import torch

def compute_iou(dist_gt, dist_pr):
    """Intersection over Union.
    
    Args:
        dist_gt (torch.Tensor): Groundtruth signed distances
        dist_pr (torch.Tensor): Predicted signed distances
    """

    occ_gt = (dist_gt < 0).byte()
    occ_pr = (dist_pr < 0).byte()

    area_union = torch.sum((occ_gt | occ_pr).float())
    area_intersect = torch.sum((occ_gt & occ_pr).float())

    iou = area_intersect / area_union
    return 100. * iou

