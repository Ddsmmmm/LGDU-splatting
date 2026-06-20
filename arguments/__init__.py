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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.progress_full_log_interval = 0
        self.line_segments_path = ""
        self.line_tracks_path = ""
        self.line_min_visible_views = 0
        self.line_min_length = 0.0
        self.line_confidence_mode = "length"
        self.line_confidence_power = 1.0
        self.line_confidence_min = 0.05
        self.line_confidence_percentile = 0.0
        self.line_tau = 0.0
        self.line_lambda_init = 0.0
        self.line_lambda_final = 0.0
        self.line_lambda_max_steps = 0
        self.line_lambda_adaptive = False
        self.line_lambda_target_ratio = 0.05
        self.line_lambda_adaptive_min = 0.0
        self.line_lambda_adaptive_max = 1.0
        self.line_loss_start_iter = 0
        self.line_loss_end_iter = 0
        self.line_loss_max_points = 2048
        self.line_loss_chunk_size = 256
        self.line_loss_point_chunk_size = 4096
        self.line_edge_support_enable = False
        self.line_edge_canny_low = 50
        self.line_edge_canny_high = 150
        self.line_edge_sigma_px = 2.0
        self.line_edge_sample_count = 16
        self.line_edge_min_valid_ratio = 0.5
        self.line_edge_min_projected_length = 4.0
        self.line_edge_chunk_size = 2048
        self.line_multiview_edge_support_enable = False
        self.line_multiview_edge_max_views = 0
        self.line_multiview_edge_min_views = 1
        self.line_photo_reweight_enable = False
        self.line_photo_lambda_init = 0.0
        self.line_photo_lambda_final = 0.0
        self.line_photo_lambda_max_steps = 0
        self.line_photo_lambda_adaptive = False
        self.line_photo_lambda_target_ratio = 0.03
        self.line_photo_lambda_adaptive_min = 0.0
        self.line_photo_lambda_adaptive_max = 0.02
        self.line_photo_start_iter = 0
        self.line_photo_end_iter = 0
        self.line_photo_sample_count = 32
        self.line_photo_mask_dilation_px = 2
        self.line_photo_min_valid_ratio = 0.5
        self.line_photo_min_projected_length = 4.0
        self.line_photo_chunk_size = 2048
        self.line_photo_min_weight = 0.0
        self.line_photo_charbonnier_eps = 0.001
        self.line_image_edge_loss_enable = False
        self.line_image_edge_lambda_init = 0.0
        self.line_image_edge_lambda_final = 0.0
        self.line_image_edge_lambda_max_steps = 0
        self.line_image_edge_lambda_adaptive = False
        self.line_image_edge_lambda_target_ratio = 0.02
        self.line_image_edge_lambda_adaptive_min = 0.0
        self.line_image_edge_lambda_adaptive_max = 0.01
        self.line_image_edge_start_iter = 0
        self.line_image_edge_end_iter = 0
        self.line_image_edge_sample_count = 32
        self.line_image_edge_mask_dilation_px = 2
        self.line_image_edge_min_valid_ratio = 0.5
        self.line_image_edge_min_projected_length = 4.0
        self.line_image_edge_chunk_size = 2048
        self.line_image_edge_charbonnier_eps = 0.001
        self.line_image_edge_min_weight = 0.0
        self.coverage_loss_enable = False
        self.coverage_lambda_init = 0.0
        self.coverage_lambda_final = 0.0
        self.coverage_lambda_max_steps = 0
        self.coverage_lambda_adaptive = False
        self.coverage_lambda_target_ratio = 0.02
        self.coverage_lambda_adaptive_min = 0.0
        self.coverage_lambda_adaptive_max = 0.01
        self.coverage_loss_start_iter = 0
        self.coverage_loss_end_iter = 0
        self.coverage_loss_interval = 1
        self.coverage_alpha_target = 0.98
        self.coverage_mask_mode = "dark"
        self.coverage_dark_threshold = 0.35
        self.coverage_error_threshold = 0.03
        self.coverage_use_line_mask = False
        self.coverage_line_mask_dilation_px = 6
        self.line_orient_enable = False
        self.line_orient_lambda_init = 0.0
        self.line_orient_lambda_final = 0.0
        self.line_orient_lambda_max_steps = 0
        self.line_orient_start_iter = 0
        self.line_orient_end_iter = 0
        self.line_orient_tau = 0.0
        self.line_orient_max_points = 2048
        self.line_orient_min_weight = 0.0
        self.line_densify_enable = True
        self.line_densify_sigma = 0.0
        self.line_densify_tau_g = 0.0
        self.line_densify_s_g = 0.0
        self.line_densify_tau_alpha = 0.08
        self.line_densify_s_alpha = 0.02
        self.line_densify_tau_d = 0.0
        self.line_densify_s_d = 0.0
        self.line_densify_clone_prob_thresh = 0.55
        self.line_densify_prune_prob_thresh = 0.75
        self.line_densify_prune_start_iter = 0
        self.line_densify_prune_end_iter = 0
        self.line_densify_chunk_size = 256
        self.line_densify_point_chunk_size = 4096
        self.densify_max_points_per_stage = 2048
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
