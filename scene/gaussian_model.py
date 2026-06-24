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

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.line_utils import point_to_segment_distance

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.filter_3D = torch.empty(0)
        self.mip_filter_active = False
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.filter_3D,
            self.mip_filter_active,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 14:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.filter_3D,
            self.mip_filter_active,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            self.filter_3D = torch.zeros((self._xyz.shape[0], 1), dtype=self._xyz.dtype, device=self._xyz.device)
            self.mip_filter_active = False
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def _filter_3D_for_current_points(self):
        if (
            not torch.is_tensor(self.filter_3D)
            or self.filter_3D.numel() == 0
            or self.filter_3D.shape[0] != self._xyz.shape[0]
        ):
            return torch.zeros((self._xyz.shape[0], 1), dtype=self._xyz.dtype, device=self._xyz.device)
        return self.filter_3D.to(device=self._xyz.device, dtype=self._xyz.dtype)

    def has_3D_filter(self):
        return (
            self.mip_filter_active
            and torch.is_tensor(self.filter_3D)
            and self.filter_3D.numel() > 0
            and self.filter_3D.shape[0] == self._xyz.shape[0]
        )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        if not self.has_3D_filter():
            return self.get_scaling
        scales = self.get_scaling
        return torch.sqrt(torch.square(scales) + torch.square(self._filter_3D_for_current_points()))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_opacity_filter_coef(self):
        if not self.has_3D_filter():
            return torch.ones_like(self.get_opacity)
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det_before = scales_square[:, 0] * scales_square[:, 1] * scales_square[:, 2]
        filtered_square = scales_square + torch.square(self._filter_3D_for_current_points())
        det_after = filtered_square[:, 0] * filtered_square[:, 1] * filtered_square[:, 2]
        coef = torch.sqrt(det_before.clamp_min(1e-24) / det_after.clamp_min(1e-24))
        return coef[..., None]

    @property
    def get_opacity_with_3D_filter(self):
        return self.get_opacity * self.get_opacity_filter_coef()
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_with_3D_filter(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling_with_3D_filter, scaling_modifier, self._rotation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras, filter_scale=0.4472135955, screen_margin=0.15):
        if self._xyz.numel() == 0:
            self.filter_3D = torch.empty((0, 1), dtype=self._xyz.dtype, device=self._xyz.device)
            self.mip_filter_active = True
            return

        print("Computing Mip-style 3D filter")
        xyz = self.get_xyz
        distance = torch.full((xyz.shape[0],), 1e8, dtype=xyz.dtype, device=xyz.device)
        valid_points = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
        focal_length = 0.0
        margin = max(float(screen_margin), 0.0)

        for camera in cameras:
            R = torch.as_tensor(camera.R, device=xyz.device, dtype=xyz.dtype)
            T = torch.as_tensor(camera.T, device=xyz.device, dtype=xyz.dtype)
            xyz_cam = xyz @ R + T[None, :]
            z = xyz_cam[:, 2]
            z_safe = z.clamp_min(1e-3)
            focal_x = float(camera.image_width) / (2.0 * math.tan(float(camera.FoVx) * 0.5))
            focal_y = float(camera.image_height) / (2.0 * math.tan(float(camera.FoVy) * 0.5))
            x = xyz_cam[:, 0] / z_safe * focal_x + camera.image_width * 0.5
            y = xyz_cam[:, 1] / z_safe * focal_y + camera.image_height * 0.5

            valid_depth = z > 0.2
            in_screen = (
                (x >= -margin * camera.image_width)
                & (x <= (1.0 + margin) * camera.image_width)
                & (y >= -margin * camera.image_height)
                & (y <= (1.0 + margin) * camera.image_height)
            )
            valid = valid_depth & in_screen
            distance[valid] = torch.minimum(distance[valid], z[valid])
            valid_points |= valid
            focal_length = max(focal_length, focal_x)

        if valid_points.any():
            distance[~valid_points] = distance[valid_points].max()
        else:
            distance[:] = 1.0
        focal_length = max(focal_length, 1e-6)
        self.filter_3D = (distance / focal_length * float(filter_scale))[..., None]
        self.mip_filter_active = True

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.densify_max_points_per_stage = training_args.densify_max_points_per_stage
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def _cap_selected_mask(self, selected_pts_mask, scores):
        max_points = getattr(self, "densify_max_points_per_stage", 0)
        if max_points is None or max_points <= 0:
            return selected_pts_mask

        selected_count = int(selected_pts_mask.sum().item())
        if selected_count <= max_points:
            return selected_pts_mask

        selected_indices = torch.nonzero(selected_pts_mask, as_tuple=False).squeeze(1)
        selected_scores = scores[selected_indices]
        topk = torch.topk(selected_scores, k=max_points, largest=True).indices

        capped_mask = torch.zeros_like(selected_pts_mask, dtype=torch.bool)
        capped_mask[selected_indices[topk]] = True
        return capped_mask

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        filter_3D = self._filter_3D_for_current_points().detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        if self.has_3D_filter():
            current_opacity_with_filter = self.get_opacity_with_3D_filter
            opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * 0.01)
            opacities_new = opacities_new / self.get_opacity_filter_coef().clamp_min(1e-6)
            opacities_new = self.inverse_opacity_activation(opacities_new.clamp(1e-6, 0.99))
        else:
            opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        ply_property_names = {p.name for p in plydata.elements[0].properties}
        if "filter_3D" in ply_property_names:
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
        else:
            filter_3D = np.zeros((xyz.shape[0], 1), dtype=np.float32)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        self.mip_filter_active = bool(np.any(filter_3D > 0.0))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        old_filter_3D = self._filter_3D_for_current_points()
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self.filter_3D = old_filter_3D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        old_filter_3D = self._filter_3D_for_current_points()
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        new_filter_3D = torch.zeros((new_xyz.shape[0], 1), dtype=old_filter_3D.dtype, device=old_filter_3D.device)
        self.filter_3D = torch.cat((old_filter_3D, new_filter_3D), dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if hasattr(self, "_line_clone_mask") and self._line_clone_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, self._line_clone_mask)

        selected_pts_mask = self._cap_selected_mask(selected_pts_mask, padded_grad)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        grad_norms = torch.norm(grads, dim=-1)
        selected_pts_mask = torch.where(grad_norms >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if hasattr(self, "_line_clone_mask") and self._line_clone_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, self._line_clone_mask)

        selected_pts_mask = self._cap_selected_mask(selected_pts_mask, grad_norms)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def _normalize_line_confidences_for_densify(self, line_confidences):
        if line_confidences is None:
            return None
        confidences = line_confidences.clamp_min(0.0)
        positive_confidence = confidences[confidences > 0.0]
        if positive_confidence.numel() > 0:
            confidences = confidences / positive_confidence.mean().clamp_min(1e-6)
        return confidences

    def _nearest_gaussian_for_points(self, points, point_chunk_size=4096, candidate_chunk_size=256):
        point_chunk_size = max(int(point_chunk_size), 1)
        candidate_chunk_size = max(int(candidate_chunk_size), 1)
        xyz = self.get_xyz.detach()
        nearest_distances = torch.empty((points.shape[0],), device=points.device)
        nearest_indices = torch.empty((points.shape[0],), dtype=torch.long, device=points.device)

        for c_start in range(0, points.shape[0], candidate_chunk_size):
            c_end = min(c_start + candidate_chunk_size, points.shape[0])
            candidate_chunk = points[c_start:c_end]
            local_min_sq = torch.full((candidate_chunk.shape[0],), float("inf"), device=points.device)
            local_min_indices = torch.zeros((candidate_chunk.shape[0],), dtype=torch.long, device=points.device)

            for p_start in range(0, xyz.shape[0], point_chunk_size):
                p_end = min(p_start + point_chunk_size, xyz.shape[0])
                xyz_chunk = xyz[p_start:p_end]
                dist2 = ((candidate_chunk[:, None, :] - xyz_chunk[None, :, :]) ** 2).sum(dim=-1)
                chunk_min_sq, chunk_min_indices = dist2.min(dim=1)
                improved = chunk_min_sq < local_min_sq
                local_min_sq = torch.where(improved, chunk_min_sq, local_min_sq)
                local_min_indices = torch.where(improved, chunk_min_indices + p_start, local_min_indices)

            nearest_distances[c_start:c_end] = torch.sqrt(local_min_sq.clamp_min(0.0))
            nearest_indices[c_start:c_end] = local_min_indices

        return nearest_distances, nearest_indices

    def line_guided_unpool(self, line_segments, line_confidences, line_cfg, scene_extent):
        if line_segments is None or line_segments.numel() == 0:
            return 0

        samples_per_line = max(int(getattr(line_cfg, "line_unpool_samples_per_line", 4)), 1)
        max_points = max(int(getattr(line_cfg, "line_unpool_max_points", 1024)), 0)
        if max_points <= 0:
            return 0

        confidences = self._normalize_line_confidences_for_densify(line_confidences)
        if confidences is None:
            confidences = torch.ones((line_segments.shape[0],), device=line_segments.device)
        confidence_power = max(float(getattr(line_cfg, "line_densify_confidence_power", 1.0)), 0.0)
        if confidence_power != 1.0:
            confidences = confidences.clamp_min(0.0).pow(confidence_power)
        confidence_max = float(getattr(line_cfg, "line_densify_confidence_max", 3.0))
        if confidence_max > 0.0:
            confidences = confidences.clamp(max=confidence_max)

        min_confidence = max(float(getattr(line_cfg, "line_unpool_min_confidence", 0.0)), 0.0)
        valid_lines = confidences >= min_confidence
        if not valid_lines.any():
            return 0

        valid_segments = line_segments[valid_lines]
        valid_confidences = confidences[valid_lines]
        t = torch.linspace(
            0.0,
            1.0,
            samples_per_line + 2,
            device=line_segments.device,
            dtype=line_segments.dtype,
        )[1:-1]
        line_starts = valid_segments[:, 0, :]
        line_ends = valid_segments[:, 1, :]
        candidates = line_starts[:, None, :] * (1.0 - t[None, :, None]) + line_ends[:, None, :] * t[None, :, None]
        candidates = candidates.reshape(-1, 3)
        candidate_confidences = valid_confidences[:, None].repeat(1, samples_per_line).reshape(-1)

        candidate_limit = max_points * max(int(getattr(line_cfg, "line_unpool_candidate_factor", 4)), 1)
        if candidate_confidences.numel() > candidate_limit:
            keep = torch.topk(candidate_confidences, k=candidate_limit, largest=True).indices
            candidates = candidates[keep]
            candidate_confidences = candidate_confidences[keep]

        nearest_distances, nearest_indices = self._nearest_gaussian_for_points(
            candidates,
            point_chunk_size=getattr(line_cfg, "line_unpool_point_chunk_size", 4096),
            candidate_chunk_size=getattr(line_cfg, "line_unpool_chunk_size", 256),
        )

        support_radius = float(getattr(line_cfg, "line_unpool_support_radius", 0.0))
        if support_radius <= 0.0:
            support_radius = 0.005 * float(scene_extent)
        support_radius = max(support_radius, 1e-6)
        support_temperature = max(0.25 * support_radius, 1e-6)
        sparse_score = torch.sigmoid((nearest_distances - support_radius) / support_temperature)

        nearest_opacity = self.get_opacity[nearest_indices].squeeze(-1)
        alpha_target = float(getattr(line_cfg, "line_unpool_alpha_target", 0.08))
        low_alpha_temperature = max(0.25 * max(alpha_target, 1e-6), 1e-6)
        low_alpha_score = torch.sigmoid((alpha_target - nearest_opacity) / low_alpha_temperature)
        low_alpha_boost = max(float(getattr(line_cfg, "line_unpool_low_alpha_boost", 0.5)), 0.0)

        candidate_scores = candidate_confidences * sparse_score * (1.0 + low_alpha_boost * low_alpha_score)
        threshold = float(getattr(line_cfg, "line_unpool_score_threshold", 0.05))
        selected = candidate_scores >= threshold
        if not selected.any():
            return 0

        selected_indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
        selected_scores = candidate_scores[selected_indices]
        if selected_indices.numel() > max_points:
            topk = torch.topk(selected_scores, k=max_points, largest=True).indices
            selected_indices = selected_indices[topk]

        nearest_indices = nearest_indices[selected_indices]
        new_xyz = candidates[selected_indices]
        new_features_dc = self._features_dc[nearest_indices]
        new_features_rest = self._features_rest[nearest_indices]

        opacity_init = min(max(float(getattr(line_cfg, "line_unpool_opacity_init", 0.05)), 1e-4), 0.99)
        new_opacities = self.inverse_opacity_activation(
            opacity_init * torch.ones((selected_indices.shape[0], 1), dtype=self._opacity.dtype, device=self._opacity.device)
        )

        scale_factor = max(float(getattr(line_cfg, "line_unpool_scale_factor", 0.7)), 1e-3)
        new_scaling_activated = self.get_scaling[nearest_indices] * scale_factor
        new_scaling_activated = torch.minimum(
            new_scaling_activated,
            support_radius * torch.ones_like(new_scaling_activated),
        ).clamp_min(1e-5)
        new_scaling = self.scaling_inverse_activation(new_scaling_activated)
        new_rotation = self._rotation[nearest_indices]
        new_tmp_radii = torch.zeros((selected_indices.shape[0],), device="cuda")

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        return int(selected_indices.shape[0])

    def _line_guided_probs(self, grads, line_segments, line_cfg, line_confidences=None):
        xyz = self.get_xyz
        d, nearest_line_idx = point_to_segment_distance(
            xyz,
            line_segments,
            segment_chunk_size=line_cfg.line_densify_chunk_size,
            point_chunk_size=line_cfg.line_densify_point_chunk_size,
            return_indices=True,
        )
        g = torch.norm(grads, dim=-1)
        alpha = self.get_opacity.squeeze(-1)

        if line_confidences is None:
            nearest_confidence = torch.ones_like(d)
        else:
            nearest_confidence = line_confidences[nearest_line_idx].clamp_min(0.0)
            positive_confidence = nearest_confidence[nearest_confidence > 0.0]
            if positive_confidence.numel() > 0:
                nearest_confidence = nearest_confidence / positive_confidence.mean().clamp_min(1e-6)
        confidence_power = max(float(getattr(line_cfg, "line_densify_confidence_power", 1.0)), 0.0)
        if confidence_power != 1.0:
            nearest_confidence = nearest_confidence.clamp_min(0.0).pow(confidence_power)
        confidence_max = float(getattr(line_cfg, "line_densify_confidence_max", 3.0))
        if confidence_max > 0.0:
            nearest_confidence = nearest_confidence.clamp(max=confidence_max)

        def _quantile_safe(values, q):
            if values.numel() == 0:
                return torch.tensor(0.0, device=values.device)
            return torch.quantile(values, q)

        sigma = line_cfg.line_densify_sigma
        if sigma <= 0.0:
            if getattr(line_cfg, "line_tau", 0.0) > 0.0:
                sigma = float(line_cfg.line_tau)
            else:
                weighted_distances = d[nearest_confidence > 0.0]
                sigma = _quantile_safe(weighted_distances, 0.25).item()
        sigma = max(sigma, 1e-6)
        w = torch.exp(-0.5 * (d / sigma) ** 2)

        tau_g = line_cfg.line_densify_tau_g
        if tau_g <= 0.0:
            tau_g = _quantile_safe(g, 0.7).item()

        tau_d = line_cfg.line_densify_tau_d
        if tau_d <= 0.0:
            tau_d = _quantile_safe(d, 0.85).item()

        tau_alpha = line_cfg.line_densify_tau_alpha
        if tau_alpha <= 0.0:
            tau_alpha = 0.08

        s_g = line_cfg.line_densify_s_g
        if s_g <= 0.0:
            s_g = 0.1 * max(tau_g, 1e-6)
        s_g = max(s_g, 1e-6)

        s_a = line_cfg.line_densify_s_alpha
        if s_a <= 0.0:
            s_a = 0.02
        s_a = max(s_a, 1e-6)

        s_d = line_cfg.line_densify_s_d
        if s_d <= 0.0:
            s_d = 0.1 * max(tau_d, 1e-6)
        s_d = max(s_d, 1e-6)

        sig_g = torch.sigmoid((g - tau_g) / s_g)
        sig_a = torch.sigmoid((alpha - tau_alpha) / s_a)
        sig_low_a = torch.sigmoid((tau_alpha - alpha) / s_a)
        line_score = w * nearest_confidence
        low_alpha_boost = max(float(getattr(line_cfg, "line_densify_low_alpha_boost", 0.0)), 0.0)
        if low_alpha_boost > 0.0:
            line_score = line_score * (1.0 + low_alpha_boost * sig_low_a)
        p_clone = line_score * sig_g * sig_a

        sig_d = torch.sigmoid((d - tau_d) / s_d)
        sig_low_g = torch.sigmoid((tau_g - g) / s_g)
        p_prune = sig_d * sig_low_a * sig_low_g

        return line_score, p_clone, p_prune

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, line_segments=None, line_confidences=None, line_cfg=None, iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._line_clone_mask = None
        line_prune_mask = None
        densify_grads = grads
        self._last_line_unpool_count = 0
        if line_segments is not None and line_cfg is not None and line_cfg.line_densify_enable:
            mode = str(getattr(line_cfg, "line_densify_mode", "off")).lower()
            in_window = True
            if iteration is not None:
                start_iter = int(getattr(line_cfg, "line_densify_start_iter", 0))
                end_iter = int(getattr(line_cfg, "line_densify_end_iter", 0))
                in_window = iteration >= start_iter and (end_iter <= 0 or iteration <= end_iter)
            if mode != "off" and in_window:
                line_score, p_clone, p_prune = self._line_guided_probs(grads, line_segments, line_cfg, line_confidences)
                if mode in ("score", "soft", "score_boost"):
                    score_boost = max(float(getattr(line_cfg, "line_densify_score_boost", 1.0)), 0.0)
                    densify_grads = grads * (1.0 + score_boost * line_score[:, None].detach())
                elif mode in ("legacy", "legacy_mask", "mask", "hard"):
                    self._line_clone_mask = p_clone >= line_cfg.line_densify_clone_prob_thresh
                else:
                    raise ValueError(f"Unknown line_densify_mode: {mode}")
                if getattr(line_cfg, "line_densify_prune_enable", False) and iteration is not None:
                    if iteration >= line_cfg.line_densify_prune_start_iter:
                        if line_cfg.line_densify_prune_end_iter <= 0 or iteration <= line_cfg.line_densify_prune_end_iter:
                            line_prune_mask = p_prune >= line_cfg.line_densify_prune_prob_thresh

        self.tmp_radii = radii
        self.densify_and_clone(densify_grads, max_grad, extent)
        self.densify_and_split(densify_grads, max_grad, extent)
        self._line_clone_mask = None

        if line_segments is not None and line_cfg is not None and getattr(line_cfg, "line_unpool_enable", False):
            unpool_window = True
            if iteration is not None:
                start_iter = int(getattr(line_cfg, "line_unpool_start_iter", 0))
                end_iter = int(getattr(line_cfg, "line_unpool_end_iter", 0))
                interval = max(int(getattr(line_cfg, "line_unpool_interval", 500)), 1)
                unpool_window = (
                    iteration >= start_iter
                    and (end_iter <= 0 or iteration <= end_iter)
                    and iteration % interval == 0
                )
            if unpool_window:
                self._last_line_unpool_count = self.line_guided_unpool(
                    line_segments,
                    line_confidences,
                    line_cfg,
                    extent,
                )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if line_prune_mask is not None:
            prune_mask = torch.logical_or(prune_mask, line_prune_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
