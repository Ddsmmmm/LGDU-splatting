#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, build_rotation
from utils.line_utils import load_line_segments_with_confidence, point_to_segment_distance, projected_line_edge_support, projected_line_mask
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _renormalize_positive_confidences(confidences):
    positive = confidences > 0
    if not positive.any():
        return confidences
    normalized = confidences.clone()
    mean_positive = normalized[positive].mean().clamp_min(1e-8)
    normalized[positive] = normalized[positive] / mean_positive
    return normalized


def _apply_confidence_percentile(confidences, percentile):
    percentile = float(percentile)
    if percentile <= 0.0:
        return confidences, 0.0, confidences.numel()
    percentile = min(max(percentile, 0.0), 100.0)
    positive = confidences.detach() > 0.0
    if positive.any():
        threshold = torch.quantile(confidences.detach()[positive], percentile / 100.0)
    else:
        threshold = torch.tensor(float("inf"), device=confidences.device)
    keep = confidences >= threshold
    filtered = torch.where(keep, confidences, torch.zeros_like(confidences))
    filtered = _renormalize_positive_confidences(filtered)
    return filtered, threshold.item(), int(keep.sum().item())


def _select_multiview_cameras(cameras, max_views):
    if max_views is None or max_views <= 0 or len(cameras) <= max_views:
        return cameras
    if max_views == 1:
        return [cameras[len(cameras) // 2]]
    indices = [
        int(round(i * (len(cameras) - 1) / float(max_views - 1)))
        for i in range(max_views)
    ]
    return [cameras[idx] for idx in indices]


@torch.no_grad()
def _apply_multiview_edge_support(line_segments, line_confidences, train_cameras, opt):
    selected_cameras = _select_multiview_cameras(train_cameras, int(opt.line_multiview_edge_max_views))
    support_sum = torch.zeros_like(line_confidences)
    valid_count = torch.zeros_like(line_confidences)

    for camera in tqdm(selected_cameras, desc="[Line] Multi-view edge support"):
        support, valid = projected_line_edge_support(
            line_segments,
            camera,
            canny_low=opt.line_edge_canny_low,
            canny_high=opt.line_edge_canny_high,
            sigma_px=opt.line_edge_sigma_px,
            sample_count=opt.line_edge_sample_count,
            min_valid_ratio=opt.line_edge_min_valid_ratio,
            min_projected_length=opt.line_edge_min_projected_length,
            line_chunk_size=opt.line_edge_chunk_size,
            return_valid=True,
        )
        support_sum += support
        valid_count += valid.float()

    support = support_sum / valid_count.clamp_min(1.0)
    if opt.line_multiview_edge_min_views > 1:
        support = torch.where(
            valid_count >= float(opt.line_multiview_edge_min_views),
            support,
            torch.zeros_like(support),
        )

    confidences = line_confidences * support
    confidences = _renormalize_positive_confidences(confidences)
    return confidences, support, valid_count


def _sobel_luma(image):
    if image.shape[0] >= 3:
        luma = 0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]
    else:
        luma = image.mean(dim=0, keepdim=True)

    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(2, 3)

    luma = luma.unsqueeze(0)
    grad_x = F.conv2d(luma, kernel_x, padding=1)
    grad_y = F.conv2d(luma, kernel_y, padding=1)
    return grad_x.squeeze(0).squeeze(0), grad_y.squeeze(0).squeeze(0)


def _projected_line_gradient_loss(render_image, gt_image, line_mask, eps):
    mask_sum = line_mask.sum()
    if mask_sum <= 1e-8:
        return None

    render_grad_x, render_grad_y = _sobel_luma(render_image)
    with torch.no_grad():
        gt_grad_x, gt_grad_y = _sobel_luma(gt_image)
    diff_x = render_grad_x - gt_grad_x
    diff_y = render_grad_y - gt_grad_y
    charbonnier = torch.sqrt(diff_x * diff_x + diff_y * diff_y + float(eps) ** 2)
    return (line_mask * charbonnier).sum() / mask_sum.clamp_min(1e-8)


def _projected_line_photometric_loss(render_image, gt_image, line_mask, eps):
    mask_sum = line_mask.sum()
    if mask_sum <= 1e-8:
        return None

    diff = render_image - gt_image
    eps = max(float(eps), 0.0)
    if eps > 0.0:
        per_channel = torch.sqrt(diff * diff + eps * eps)
    else:
        per_channel = diff.abs()
    per_pixel = per_channel.mean(dim=0)
    return (line_mask * per_pixel).sum() / mask_sum.clamp_min(1e-8)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    line_segments = None
    line_confidences = None
    line_weight_func = None
    line_photo_weight_func = None
    line_image_edge_weight_func = None
    line_orient_weight_func = None
    line_loss_active = False
    line_photo_active = False
    line_image_edge_active = False
    line_edge_support_active = False
    line_orient_active = False
    line_densify_active = False
    if opt.line_segments_path or opt.line_tracks_path:
        line_segments, line_confidences = load_line_segments_with_confidence(
            obj_path=opt.line_segments_path,
            tracks_path=opt.line_tracks_path,
            device="cuda",
            min_visible_views=opt.line_min_visible_views,
            min_length=opt.line_min_length,
            confidence_mode=opt.line_confidence_mode,
            confidence_power=opt.line_confidence_power,
            confidence_min=opt.line_confidence_min,
        )
        print(
            "[Line] Loaded {} segments. confidence min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
                line_segments.shape[0],
                line_confidences.min().item(),
                line_confidences.mean().item(),
                line_confidences.max().item(),
            )
        )
        if opt.line_multiview_edge_support_enable:
            line_confidences, multiview_support, multiview_valid_count = _apply_multiview_edge_support(
                line_segments,
                line_confidences,
                scene.getTrainCameras(),
                opt,
            )
            print(
                "[Line] Multi-view edge support min/mean/max: {:.4f}/{:.4f}/{:.4f}; valid views mean/max: {:.2f}/{:.0f}".format(
                    multiview_support.min().item(),
                    multiview_support.mean().item(),
                    multiview_support.max().item(),
                    multiview_valid_count.mean().item(),
                    multiview_valid_count.max().item(),
                )
            )
        if opt.line_confidence_percentile > 0.0:
            line_confidences, threshold, kept_count = _apply_confidence_percentile(
                line_confidences,
                opt.line_confidence_percentile,
            )
            print(
                "[Line] Confidence percentile {:.1f}: threshold {:.4f}, kept {}/{} segments".format(
                    opt.line_confidence_percentile,
                    threshold,
                    kept_count,
                    line_confidences.shape[0],
                )
            )
        print(
            "[Line] Final confidence min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
                line_confidences.min().item(),
                line_confidences.mean().item(),
                line_confidences.max().item(),
            )
        )
        if opt.line_lambda_max_steps > 0:
            line_weight_func = get_expon_lr_func(opt.line_lambda_init, opt.line_lambda_final, max_steps=opt.line_lambda_max_steps)
        else:
            line_weight_func = lambda _it: opt.line_lambda_init
        line_loss_active = opt.line_tau > 0.0 and (opt.line_lambda_init > 0.0 or opt.line_lambda_adaptive)
        if opt.line_photo_lambda_max_steps > 0:
            line_photo_weight_func = get_expon_lr_func(
                opt.line_photo_lambda_init,
                opt.line_photo_lambda_final,
                max_steps=opt.line_photo_lambda_max_steps,
            )
        else:
            line_photo_weight_func = lambda _it: opt.line_photo_lambda_init
        line_photo_active = (
            opt.line_photo_reweight_enable
            and (
                opt.line_photo_lambda_init > 0.0
                or opt.line_photo_lambda_final > 0.0
                or opt.line_photo_lambda_adaptive
            )
        )
        if opt.line_image_edge_lambda_max_steps > 0:
            line_image_edge_weight_func = get_expon_lr_func(
                opt.line_image_edge_lambda_init,
                opt.line_image_edge_lambda_final,
                max_steps=opt.line_image_edge_lambda_max_steps,
            )
        else:
            line_image_edge_weight_func = lambda _it: opt.line_image_edge_lambda_init
        line_image_edge_active = (
            opt.line_image_edge_loss_enable
            and (
                opt.line_image_edge_lambda_init > 0.0
                or opt.line_image_edge_lambda_final > 0.0
                or opt.line_image_edge_lambda_adaptive
            )
        )
        line_edge_support_active = opt.line_edge_support_enable or ("edge" in opt.line_confidence_mode) or line_photo_active or line_image_edge_active
        if opt.line_orient_lambda_max_steps > 0:
            line_orient_weight_func = get_expon_lr_func(
                opt.line_orient_lambda_init,
                opt.line_orient_lambda_final,
                max_steps=opt.line_orient_lambda_max_steps,
            )
        else:
            line_orient_weight_func = lambda _it: opt.line_orient_lambda_init
        line_orient_active = (
            opt.line_orient_enable
            and (opt.line_orient_lambda_init > 0.0 or opt.line_orient_lambda_final > 0.0)
        )
        line_densify_active = opt.line_densify_enable
        if opt.line_densify_enable and opt.line_densify_prune_start_iter <= 0:
            opt.line_densify_prune_start_iter = int(0.6 * opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_line_loss_for_log = 0.0
    ema_line_weight_for_log = 0.0
    ema_line_photo_loss_for_log = 0.0
    ema_line_photo_weight_for_log = 0.0
    ema_line_image_edge_loss_for_log = 0.0
    ema_line_image_edge_weight_for_log = 0.0
    ema_orient_loss_for_log = 0.0
    ema_orient_weight_for_log = 0.0
    ema_edge_support_for_log = 0.0
    ema_line_mask_mean_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        photometric_loss = loss

        line_loss_value = 0.0
        line_weight_value = 0.0
        line_photo_loss_value = 0.0
        line_photo_weight_value = 0.0
        line_image_edge_loss_value = 0.0
        line_image_edge_weight_value = 0.0
        orient_loss_value = 0.0
        orient_weight_value = 0.0
        edge_support_mean = 0.0
        line_mask_mean = 0.0

        center_loss_window = (
            line_loss_active
            and iteration >= opt.line_loss_start_iter
            and (opt.line_loss_end_iter <= 0 or iteration <= opt.line_loss_end_iter)
        )
        photo_loss_window = (
            line_photo_active
            and iteration >= opt.line_photo_start_iter
            and (opt.line_photo_end_iter <= 0 or iteration <= opt.line_photo_end_iter)
        )
        image_edge_loss_window = (
            line_image_edge_active
            and iteration >= opt.line_image_edge_start_iter
            and (opt.line_image_edge_end_iter <= 0 or iteration <= opt.line_image_edge_end_iter)
        )
        orient_loss_window = (
            line_orient_active
            and iteration >= opt.line_orient_start_iter
            and (opt.line_orient_end_iter <= 0 or iteration <= opt.line_orient_end_iter)
        )

        line_view_confidences = line_confidences
        if line_segments is not None and line_edge_support_active and (center_loss_window or photo_loss_window or image_edge_loss_window or orient_loss_window):
            edge_support = projected_line_edge_support(
                line_segments,
                viewpoint_cam,
                canny_low=opt.line_edge_canny_low,
                canny_high=opt.line_edge_canny_high,
                sigma_px=opt.line_edge_sigma_px,
                sample_count=opt.line_edge_sample_count,
                min_valid_ratio=opt.line_edge_min_valid_ratio,
                min_projected_length=opt.line_edge_min_projected_length,
                line_chunk_size=opt.line_edge_chunk_size,
            )
            line_view_confidences = line_confidences * edge_support
            edge_support_mean = edge_support.mean().item()

        line_mask = None
        if line_segments is not None and (photo_loss_window or image_edge_loss_window):
            if photo_loss_window:
                mask_sample_count = opt.line_photo_sample_count
                mask_dilation_px = opt.line_photo_mask_dilation_px
                mask_min_valid_ratio = opt.line_photo_min_valid_ratio
                mask_min_projected_length = opt.line_photo_min_projected_length
                mask_chunk_size = opt.line_photo_chunk_size
                mask_min_weight = opt.line_photo_min_weight
            else:
                mask_sample_count = opt.line_image_edge_sample_count
                mask_dilation_px = opt.line_image_edge_mask_dilation_px
                mask_min_valid_ratio = opt.line_image_edge_min_valid_ratio
                mask_min_projected_length = opt.line_image_edge_min_projected_length
                mask_chunk_size = opt.line_image_edge_chunk_size
                mask_min_weight = opt.line_image_edge_min_weight
            line_mask = projected_line_mask(
                line_segments,
                viewpoint_cam,
                confidences=line_view_confidences,
                sample_count=mask_sample_count,
                dilation_px=mask_dilation_px,
                min_valid_ratio=mask_min_valid_ratio,
                min_projected_length=mask_min_projected_length,
                line_chunk_size=mask_chunk_size,
                min_weight=mask_min_weight,
            )
            line_mask_mean = line_mask.mean().item()

        if line_mask is not None and photo_loss_window:
            line_photo_loss_value = _projected_line_photometric_loss(
                image,
                gt_image,
                line_mask,
                opt.line_photo_charbonnier_eps,
            )
            if line_photo_loss_value is not None and opt.line_photo_lambda_adaptive:
                photo_weight = (
                    opt.line_photo_lambda_target_ratio
                    * photometric_loss.detach()
                    / line_photo_loss_value.detach().clamp_min(1e-12)
                )
                if opt.line_photo_lambda_adaptive_max > 0.0:
                    photo_weight = photo_weight.clamp(max=opt.line_photo_lambda_adaptive_max)
                photo_weight = photo_weight.clamp(min=opt.line_photo_lambda_adaptive_min)
            else:
                photo_weight = line_photo_weight_func(iteration)
            if line_photo_loss_value is not None:
                line_photo_weight_value = photo_weight.item() if torch.is_tensor(photo_weight) else float(photo_weight)
                loss = loss + photo_weight * line_photo_loss_value
            else:
                line_photo_loss_value = 0.0

        if line_mask is not None and image_edge_loss_window:
            line_image_edge_loss_value = _projected_line_gradient_loss(
                image,
                gt_image,
                line_mask,
                opt.line_image_edge_charbonnier_eps,
            )
            if line_image_edge_loss_value is not None and opt.line_image_edge_lambda_adaptive:
                image_edge_weight = (
                    opt.line_image_edge_lambda_target_ratio
                    * photometric_loss.detach()
                    / line_image_edge_loss_value.detach().clamp_min(1e-12)
                )
                if opt.line_image_edge_lambda_adaptive_max > 0.0:
                    image_edge_weight = image_edge_weight.clamp(max=opt.line_image_edge_lambda_adaptive_max)
                image_edge_weight = image_edge_weight.clamp(min=opt.line_image_edge_lambda_adaptive_min)
            else:
                image_edge_weight = line_image_edge_weight_func(iteration)
            if line_image_edge_loss_value is not None:
                line_image_edge_weight_value = image_edge_weight.item() if torch.is_tensor(image_edge_weight) else float(image_edge_weight)
                loss = loss + image_edge_weight * line_image_edge_loss_value
            else:
                line_image_edge_loss_value = 0.0

        sampled_xyz = None
        sampled_point_indices = None
        sampled_distances = None
        sampled_nearest_line_idx = None
        if line_segments is not None and (center_loss_window or orient_loss_window):
            xyz_full = gaussians.get_xyz
            max_points = 0
            if center_loss_window:
                max_points = max(max_points, int(opt.line_loss_max_points))
            if orient_loss_window:
                max_points = max(max_points, int(opt.line_orient_max_points))
            if max_points > 0 and xyz_full.shape[0] > max_points:
                sampled_point_indices = torch.randperm(xyz_full.shape[0], device=xyz_full.device)[:max_points]
                sampled_xyz = xyz_full[sampled_point_indices]
            else:
                sampled_point_indices = torch.arange(xyz_full.shape[0], device=xyz_full.device)
                sampled_xyz = xyz_full

            if center_loss_window:
                sampled_distances, sampled_nearest_line_idx = point_to_segment_distance(
                    sampled_xyz,
                    line_segments,
                    segment_chunk_size=opt.line_loss_chunk_size,
                    point_chunk_size=opt.line_loss_point_chunk_size,
                    return_indices=True,
                )
            else:
                with torch.no_grad():
                    sampled_distances, sampled_nearest_line_idx = point_to_segment_distance(
                        sampled_xyz,
                        line_segments,
                        segment_chunk_size=opt.line_loss_chunk_size,
                        point_chunk_size=opt.line_loss_point_chunk_size,
                        return_indices=True,
                    )

        if center_loss_window and sampled_distances is not None:
            scheduled_line_weight = line_weight_func(iteration)
            d = sampled_distances
            if opt.line_tau > 0.0:
                tau = torch.tensor(opt.line_tau, device=d.device)
                d = torch.minimum(d, tau)
            nearest_confidence = line_view_confidences[sampled_nearest_line_idx]
            if line_edge_support_active:
                confidence_sum = nearest_confidence.sum()
                if confidence_sum > 1e-8:
                    line_loss_value = (nearest_confidence * d * d).sum() / confidence_sum.clamp_min(1e-8)
                else:
                    line_loss_value = None
            else:
                line_loss_value = (nearest_confidence * d * d).mean()
            if line_loss_value is not None and opt.line_lambda_adaptive:
                target_weight = (
                    opt.line_lambda_target_ratio
                    * photometric_loss.detach()
                    / line_loss_value.detach().clamp_min(1e-12)
                )
                if opt.line_lambda_adaptive_max > 0.0:
                    target_weight = target_weight.clamp(max=opt.line_lambda_adaptive_max)
                line_weight = target_weight.clamp(min=opt.line_lambda_adaptive_min)
            else:
                line_weight = scheduled_line_weight
            if line_loss_value is not None:
                line_weight_value = line_weight.item() if torch.is_tensor(line_weight) else float(line_weight)
                loss = loss + line_weight * line_loss_value
            else:
                line_loss_value = 0.0

        if orient_loss_window and sampled_distances is not None:
            orient_tau = opt.line_orient_tau if opt.line_orient_tau > 0.0 else opt.line_tau
            if orient_tau > 0.0:
                d_detached = sampled_distances.detach()
                nearest_line_idx = sampled_nearest_line_idx.detach()
                orient_weights = line_view_confidences[nearest_line_idx].detach()
                near_mask = d_detached <= orient_tau
                if opt.line_orient_min_weight > 0.0:
                    near_mask = torch.logical_and(near_mask, orient_weights >= opt.line_orient_min_weight)
                if near_mask.any():
                    selected_point_indices = sampled_point_indices[near_mask]
                    selected_line_indices = nearest_line_idx[near_mask]
                    selected_distances = d_detached[near_mask]
                    selected_weights = orient_weights[near_mask]

                    scales = gaussians.get_scaling[selected_point_indices]
                    axis_indices = scales.detach().argmax(dim=1)
                    rotations = build_rotation(gaussians.get_rotation[selected_point_indices])
                    batch_indices = torch.arange(rotations.shape[0], device=rotations.device)
                    major_axes = rotations[batch_indices, :, axis_indices]
                    major_axes = torch.nn.functional.normalize(major_axes, dim=-1)

                    nearest_segments = line_segments[selected_line_indices].detach()
                    line_dirs = nearest_segments[:, 1, :] - nearest_segments[:, 0, :]
                    line_dirs = torch.nn.functional.normalize(line_dirs, dim=-1)

                    alignment = torch.abs((major_axes * line_dirs).sum(dim=-1)).clamp(0.0, 1.0)
                    orient_error = 1.0 - alignment * alignment
                    proximity_weight = torch.exp(-0.5 * (selected_distances / max(orient_tau, 1e-6)) ** 2)
                    final_weights = selected_weights * proximity_weight
                    weight_sum = final_weights.sum()
                    if weight_sum > 1e-8:
                        orient_loss_value = (final_weights * orient_error).sum() / weight_sum.clamp_min(1e-8)
                        orient_weight = line_orient_weight_func(iteration)
                        orient_weight_value = orient_weight.item() if torch.is_tensor(orient_weight) else float(orient_weight)
                        loss = loss + orient_weight * orient_loss_value

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            if torch.is_tensor(line_loss_value):
                line_loss_for_log = line_loss_value.item()
            else:
                line_loss_for_log = float(line_loss_value)
            ema_line_loss_for_log = 0.4 * line_loss_for_log + 0.6 * ema_line_loss_for_log
            ema_line_weight_for_log = 0.4 * line_weight_value + 0.6 * ema_line_weight_for_log
            if torch.is_tensor(line_photo_loss_value):
                line_photo_loss_for_log = line_photo_loss_value.item()
            else:
                line_photo_loss_for_log = float(line_photo_loss_value)
            ema_line_photo_loss_for_log = 0.4 * line_photo_loss_for_log + 0.6 * ema_line_photo_loss_for_log
            ema_line_photo_weight_for_log = 0.4 * line_photo_weight_value + 0.6 * ema_line_photo_weight_for_log
            if torch.is_tensor(line_image_edge_loss_value):
                line_image_edge_loss_for_log = line_image_edge_loss_value.item()
            else:
                line_image_edge_loss_for_log = float(line_image_edge_loss_value)
            ema_line_image_edge_loss_for_log = 0.4 * line_image_edge_loss_for_log + 0.6 * ema_line_image_edge_loss_for_log
            ema_line_image_edge_weight_for_log = 0.4 * line_image_edge_weight_value + 0.6 * ema_line_image_edge_weight_for_log
            if torch.is_tensor(orient_loss_value):
                orient_loss_for_log = orient_loss_value.item()
            else:
                orient_loss_for_log = float(orient_loss_value)
            ema_orient_loss_for_log = 0.4 * orient_loss_for_log + 0.6 * ema_orient_loss_for_log
            ema_orient_weight_for_log = 0.4 * orient_weight_value + 0.6 * ema_orient_weight_for_log
            ema_edge_support_for_log = 0.4 * edge_support_mean + 0.6 * ema_edge_support_for_log
            ema_line_mask_mean_for_log = 0.4 * line_mask_mean + 0.6 * ema_line_mask_mean_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "Line Loss": f"{ema_line_loss_for_log:.{7}f}",
                    "Line W": f"{ema_line_weight_for_log:.{7}f}",
                    "LinePhoto Loss": f"{ema_line_photo_loss_for_log:.{7}f}",
                    "LinePhoto W": f"{ema_line_photo_weight_for_log:.{7}f}",
                    "ImgEdge Loss": f"{ema_line_image_edge_loss_for_log:.{7}f}",
                    "ImgEdge W": f"{ema_line_image_edge_weight_for_log:.{7}f}",
                    "Orient Loss": f"{ema_orient_loss_for_log:.{7}f}",
                    "Orient W": f"{ema_orient_weight_for_log:.{7}f}",
                    "Edge Sup": f"{ema_edge_support_for_log:.{7}f}",
                    "LineMask": f"{ema_line_mask_mean_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if opt.progress_full_log_interval > 0 and iteration % opt.progress_full_log_interval == 0:
                progress_bar.write(
                    "[ITER {}] Loss={:.7f} Depth={:.7f} LineLoss={:.7f} LineW={:.7f} "
                    "LinePhotoLoss={:.7f} LinePhotoW={:.7f} ImgEdgeLoss={:.7f} ImgEdgeW={:.7f} "
                    "OrientLoss={:.7f} OrientW={:.7f} EdgeSup={:.7f} LineMask={:.7f}".format(
                        iteration,
                        ema_loss_for_log,
                        ema_Ll1depth_for_log,
                        ema_line_loss_for_log,
                        ema_line_weight_for_log,
                        ema_line_photo_loss_for_log,
                        ema_line_photo_weight_for_log,
                        ema_line_image_edge_loss_for_log,
                        ema_line_image_edge_weight_for_log,
                        ema_orient_loss_for_log,
                        ema_orient_weight_for_log,
                        ema_edge_support_for_log,
                        ema_line_mask_mean_for_log,
                    )
                )
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        line_segments=line_segments if line_densify_active else None,
                        line_cfg=opt,
                        iteration=iteration,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
