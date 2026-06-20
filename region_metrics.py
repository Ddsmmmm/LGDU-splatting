"""Local region metrics and crop exporter for rendered 3DGS-style results.

The script evaluates all methods on the same masks.  The hole mask is derived
from a reference render, usually the vanilla 3DGS baseline, so the comparison
answers whether a method improves exactly those problematic pixels.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import re

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.loss_utils import ssim


def load_image(path):
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()


def save_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0.0, 1.0).cpu().squeeze(0)
    tf.to_pil_image(image).save(path)


def safe_name(path):
    name = Path(path).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "model"


def luma(image):
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_magnitude(image):
    gray = luma(image)
    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(2, 3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y).squeeze(0).squeeze(0)


def dilate_mask(mask, radius):
    radius = max(int(radius), 0)
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask[None, None].float(), kernel_size=kernel_size, stride=1, padding=radius)[0, 0] > 0


def build_masks(gt, reference_render, args):
    gt_luma = luma(gt).squeeze(0).squeeze(0)
    ref_luma = luma(reference_render).squeeze(0).squeeze(0)
    photo_error = (reference_render - gt).abs().mean(dim=1).squeeze(0)

    dark = gt_luma <= args.dark_threshold

    edge_mag = sobel_magnitude(gt)
    positive_edges = edge_mag[edge_mag > 0]
    if positive_edges.numel() > 0:
        edge_threshold = torch.quantile(positive_edges, args.edge_percentile / 100.0)
        edge = edge_mag >= edge_threshold
    else:
        edge = torch.zeros_like(edge_mag, dtype=torch.bool)
    edge = dilate_mask(edge, args.edge_dilation)

    bright_ref = ref_luma >= args.hole_bright_threshold
    high_error = photo_error >= args.hole_error_threshold
    hole = dark & bright_ref & high_error
    hole = dilate_mask(hole, args.hole_dilation)

    return {
        "dark": dark,
        "edge": edge,
        "hole": hole,
    }


def mask_bbox(mask, pad=0):
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    height, width = mask.shape
    x0 = max(int(xs.min().item()) - pad, 0)
    y0 = max(int(ys.min().item()) - pad, 0)
    x1 = min(int(xs.max().item()) + pad + 1, width)
    y1 = min(int(ys.max().item()) + pad + 1, height)
    return x0, y0, x1, y1


def expand_bbox_to_min_size(bbox, image_shape, min_size):
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    min_size = max(int(min_size), 1)

    current_width = x1 - x0
    if current_width < min_size:
        grow = min_size - current_width
        x0 -= grow // 2
        x1 += grow - grow // 2
    current_height = y1 - y0
    if current_height < min_size:
        grow = min_size - current_height
        y0 -= grow // 2
        y1 += grow - grow // 2

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height

    return max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)


def centered_square_bbox(mask, crop_size):
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    height, width = mask.shape
    cx = int(torch.round(xs.float().mean()).item())
    cy = int(torch.round(ys.float().mean()).item())
    crop_size = max(int(crop_size), 1)
    half = crop_size // 2
    x0 = min(max(cx - half, 0), max(width - crop_size, 0))
    y0 = min(max(cy - half, 0), max(height - crop_size, 0))
    x1 = min(x0 + crop_size, width)
    y1 = min(y0 + crop_size, height)
    return x0, y0, x1, y1


def crop_tensor(image, bbox):
    x0, y0, x1, y1 = bbox
    return image[..., y0:y1, x0:x1]


def masked_psnr(render, gt, mask):
    mask_sum = mask.float().sum()
    if mask_sum <= 0:
        return None
    diff = (render - gt).squeeze(0)
    mse = (diff * diff * mask[None]).sum() / (mask_sum * 3.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def focused_pair(render, gt, mask, pad, min_metric_size):
    bbox = mask_bbox(mask, pad=pad)
    if bbox is None:
        return None, None
    bbox = expand_bbox_to_min_size(bbox, mask.shape, min_size=min_metric_size)
    crop_render = crop_tensor(render, bbox)
    crop_gt = crop_tensor(gt, bbox)
    crop_mask = crop_tensor(mask[None, None].float(), bbox)
    focused_render = crop_render * crop_mask + crop_gt * (1.0 - crop_mask)
    return focused_render, crop_gt


def evaluate_pair(render, gt, mask, metric_pad, min_metric_size):
    if mask.float().sum() <= 0:
        return None
    render_focus, gt_focus = focused_pair(render, gt, mask, metric_pad, min_metric_size)
    if render_focus is None:
        return None
    return {
        "PSNR": masked_psnr(render, gt, mask).detach().cpu().item(),
        "SSIM": ssim(render_focus, gt_focus).detach().cpu().item(),
        "LPIPS": lpips(render_focus, gt_focus, net_type="vgg").detach().cpu().item(),
        "mask_ratio": mask.float().mean().detach().cpu().item(),
    }


def list_method_dirs(model_path, split):
    split_dir = Path(model_path) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    method_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
    if not method_dirs:
        raise FileNotFoundError(f"No method directories found under {split_dir}")
    return method_dirs


def collect_render_paths(model_paths, split):
    collected = []
    for model_path in model_paths:
        method_dirs = list_method_dirs(model_path, split)
        for method_dir in method_dirs:
            renders_dir = method_dir / "renders"
            gt_dir = method_dir / "gt"
            if renders_dir.exists() and gt_dir.exists():
                collected.append(
                    {
                        "model_path": Path(model_path),
                        "method": method_dir.name,
                        "renders_dir": renders_dir,
                        "gt_dir": gt_dir,
                        "label": safe_name(model_path),
                    }
                )
    if not collected:
        raise FileNotFoundError("No render/gt directories found for the requested models.")
    return collected


def summarize(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": tensor.mean().item(),
        "median": tensor.median().item(),
        "count": int(tensor.numel()),
    }


def export_crops(crop_records, models, reference_model, args):
    if args.crop_export_dir is None or args.num_crops <= 0:
        return

    export_root = Path(args.crop_export_dir)
    selected = sorted(crop_records, key=lambda item: item["score"], reverse=True)[: args.num_crops]
    for rank, record in enumerate(selected, start=1):
        name = record["name"]
        bbox = record["bbox"]
        gt = load_image(reference_model["gt_dir"] / name)
        reference_render = load_image(reference_model["renders_dir"] / name)
        masks = build_masks(gt, reference_render, args)
        mask = masks[record["region"]]
        view_stem = Path(name).stem
        crop_dir = export_root / f"{rank:03d}_{view_stem}_{record['region']}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        save_image(crop_tensor(gt, bbox), crop_dir / "gt.png")
        mask_crop = crop_tensor(mask[None, None].float(), bbox).repeat(1, 3, 1, 1)
        save_image(mask_crop, crop_dir / "mask.png")

        for model in models:
            render_path = model["renders_dir"] / name
            if not render_path.exists():
                continue
            render = load_image(render_path)
            render_crop = crop_tensor(render, bbox)
            gt_crop = crop_tensor(gt, bbox)
            error = (render_crop - gt_crop).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            save_image(render_crop, crop_dir / f"{model['label']}.png")
            save_image(error, crop_dir / f"{model['label']}_error.png")
            del render
        meta = {
            "image": name,
            "region": record["region"],
            "score": record["score"],
            "bbox": list(map(int, bbox)),
        }
        with open(crop_dir / "meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)
        del gt, reference_render


def evaluate(model_paths, split, args):
    models = collect_render_paths(model_paths, split)
    reference_model = None
    reference_path = Path(args.hole_reference_model).resolve() if args.hole_reference_model else models[0]["model_path"].resolve()
    for model in models:
        if model["model_path"].resolve() == reference_path:
            reference_model = model
            break
    if reference_model is None:
        raise ValueError(f"Hole reference model not found in model list: {reference_path}")

    image_names = sorted(p.name for p in reference_model["renders_dir"].iterdir() if p.is_file())
    if not image_names:
        raise FileNotFoundError(f"No images found in {reference_model['renders_dir']}")

    region_names = args.regions
    metrics = {
        model["label"]: {
            region: {"PSNR": [], "SSIM": [], "LPIPS": [], "mask_ratio": []}
            for region in region_names
        }
        for model in models
    }
    per_view = {
        model["label"]: {region: {} for region in region_names}
        for model in models
    }
    mask_stats = {region: [] for region in region_names}
    crop_records = []

    print(f"Hole/reference mask model: {reference_model['model_path']}")
    for name in tqdm(image_names, desc="Region metric progress"):
        gt_path = reference_model["gt_dir"] / name
        ref_path = reference_model["renders_dir"] / name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT image: {gt_path}")
        gt = load_image(gt_path)
        reference_render = load_image(ref_path)
        masks = build_masks(gt, reference_render, args)

        for region in region_names:
            mask = masks[region]
            mask_stats[region].append(mask.float().mean().detach().cpu().item())
            if region == args.crop_region and mask.float().sum() > 0:
                bbox = centered_square_bbox(mask, args.crop_size)
                score = masked_psnr(reference_render, gt, mask)
                if bbox is not None and score is not None:
                    crop_records.append(
                        {
                            "name": name,
                            "region": region,
                            "score": -float(score.detach().cpu().item()),
                            "bbox": bbox,
                        }
                    )

        for model in models:
            render_path = model["renders_dir"] / name
            if not render_path.exists():
                raise FileNotFoundError(f"Missing render image: {render_path}")
            render = load_image(render_path)
            for region in region_names:
                result = evaluate_pair(render, gt, masks[region], args.metric_pad, args.min_metric_size)
                if result is None:
                    continue
                for key, value in result.items():
                    metrics[model["label"]][region][key].append(value)
                per_view[model["label"]][region][name] = result
            del render

        del gt, reference_render
        torch.cuda.empty_cache()

    summary = {}
    for model in models:
        label = model["label"]
        summary[label] = {}
        print(f"\nModel: {label}")
        for region in region_names:
            region_summary = {}
            for metric_name in ("PSNR", "SSIM", "LPIPS", "mask_ratio"):
                region_summary[metric_name] = summarize(metrics[label][region][metric_name])
            summary[label][region] = region_summary
            if region_summary["PSNR"] is None:
                print(f"  {region}: no valid pixels")
                continue
            print(
                "  {region:<5} PSNR {psnr:>9.4f} | SSIM {ssim_v:>9.5f} | LPIPS {lpips_v:>9.5f} | mask {mask:>8.5f}".format(
                    region=region,
                    psnr=region_summary["PSNR"]["mean"],
                    ssim_v=region_summary["SSIM"]["mean"],
                    lpips_v=region_summary["LPIPS"]["mean"],
                    mask=region_summary["mask_ratio"]["mean"],
                )
            )

    output_dir = Path(args.output_dir) if args.output_dir else Path(model_paths[-1])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"region_results_{split}.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    with open(output_dir / f"region_per_view_{split}.json", "w") as fp:
        json.dump(per_view, fp, indent=2)
    with open(output_dir / f"region_mask_stats_{split}.json", "w") as fp:
        json.dump({key: summarize(value) for key, value in mask_stats.items()}, fp, indent=2)

    export_crops(crop_records, models, reference_model, args)
    if args.crop_export_dir:
        print(f"\nCrops exported to: {args.crop_export_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate local dark/edge/hole regions and export comparison crops.")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--regions", nargs="+", default=["dark", "edge", "hole"], choices=["dark", "edge", "hole"])
    parser.add_argument("--hole_reference_model", default="", type=str)
    parser.add_argument("--dark_threshold", default=0.35, type=float)
    parser.add_argument("--edge_percentile", default=90.0, type=float)
    parser.add_argument("--edge_dilation", default=2, type=int)
    parser.add_argument("--hole_bright_threshold", default=0.65, type=float)
    parser.add_argument("--hole_error_threshold", default=0.18, type=float)
    parser.add_argument("--hole_dilation", default=2, type=int)
    parser.add_argument("--metric_pad", default=12, type=int)
    parser.add_argument("--min_metric_size", default=64, type=int)
    parser.add_argument("--crop_export_dir", default="", type=str)
    parser.add_argument("--num_crops", default=0, type=int)
    parser.add_argument("--crop_region", default="hole", choices=["dark", "edge", "hole"])
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    args = parser.parse_args()

    if not args.crop_export_dir:
        args.crop_export_dir = None
    if not args.output_dir:
        args.output_dir = None

    torch.cuda.set_device(torch.device("cuda:0"))
    evaluate(args.model_paths, args.split, args)
