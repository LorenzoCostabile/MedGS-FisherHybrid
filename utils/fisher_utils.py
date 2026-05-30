import numpy as np
import torch

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def _sample_indices(num_items: int, max_items: int) -> np.ndarray:
    if num_items <= 0:
        return np.array([], dtype=np.int64)
    if num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)
    return np.random.choice(num_items, size=max_items, replace=False)


def compute_joint_fisher_proxy(
    gaussians,
    cams_img,
    cams_seg,
    pipe,
    bg,
    random_background: bool,
    max_views: int,
    lambda_img: float,
    lambda_seg: float,
):
    if len(cams_img) == 0 or len(cams_seg) == 0:
        raise RuntimeError("Fisher hybrid requires held-out test cameras for both image and segmentation.")
    if len(cams_img) != len(cams_seg):
        raise RuntimeError("Image and segmentation held-out camera lists must have the same length.")

    sampled = _sample_indices(len(cams_img), max_views)
    num_gaussians = gaussians.get_xyz.shape[0]
    fisher_xyz = torch.zeros(num_gaussians, device="cuda")
    fisher_deform = torch.zeros(num_gaussians, device="cuda")

    params = [gaussians._xyz, gaussians.m, gaussians.sigma, gaussians._w1]
    for idx in sampled:
        cam_img = cams_img[int(idx)]
        cam_seg = cams_seg[int(idx)]

        gt_img = cam_img.get_image(bg, random_background).cuda()
        gt_seg = cam_seg.get_image(bg, random_background).cuda()

        render_img = render(cam_img, gaussians, pipe, bg, train=False, seg=False, head="img")["render"]
        render_seg = render(cam_seg, gaussians, pipe, bg, train=False, seg=True, head="seg")["render"]

        loss_img = 2.0 * l1_loss(render_img, gt_img) + 0.25 * (1.0 - ssim(render_img, gt_img))
        loss_seg = 2.0 * l1_loss(render_seg, gt_seg)
        fisher_loss = lambda_img * loss_img + lambda_seg * loss_seg

        grads = torch.autograd.grad(fisher_loss, params, retain_graph=False, create_graph=False)
        fisher_xyz += grads[0].detach().pow(2).sum(dim=1)
        fisher_deform += grads[1].detach().pow(2).sum(dim=1)
        fisher_deform += grads[2].detach().pow(2).sum(dim=(1, 2))
        fisher_deform += grads[3].detach().pow(2).sum(dim=1)

    return {
        "sampled_indices": sampled.tolist(),
        "xyz": fisher_xyz,
        "deform": fisher_deform,
    }
