from argparse import ArgumentParser
from pathlib import Path
import json

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def load_image(path):
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0)[:, :3, :, :].cuda()


def evaluate(model_paths, split):
    for scene_dir in model_paths:
        scene_path = Path(scene_dir)
        split_dir = scene_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        scene_results = {}
        per_view_results = {}

        print(f"\nScene: {scene_path}")
        for method_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            renders_dir = method_dir / "renders"
            gt_dir = method_dir / "gt"
            if not renders_dir.exists() or not gt_dir.exists():
                print(f"Skipping {method}: missing renders/gt directory")
                continue

            image_names = sorted(p.name for p in renders_dir.iterdir() if p.is_file())
            if not image_names:
                print(f"Skipping {method}: no render images found")
                continue

            ssims = []
            psnrs = []
            lpipss = []

            print(f"Method: {method}")
            for name in tqdm(image_names, desc="Metric evaluation progress"):
                render_path = renders_dir / name
                gt_path = gt_dir / name
                if not gt_path.exists():
                    raise FileNotFoundError(f"Missing GT image for {render_path}: {gt_path}")

                render = load_image(render_path)
                gt = load_image(gt_path)

                ssims.append(ssim(render, gt).detach().cpu())
                psnrs.append(psnr(render, gt).detach().cpu())
                lpipss.append(lpips(render, gt, net_type="vgg").detach().cpu())

                del render, gt
                torch.cuda.empty_cache()

            ssims_t = torch.tensor(ssims)
            psnrs_t = torch.tensor(psnrs)
            lpipss_t = torch.tensor(lpipss)

            print("  SSIM : {:>12.7f}".format(ssims_t.mean()))
            print("  PSNR : {:>12.7f}".format(psnrs_t.mean()))
            print("  LPIPS: {:>12.7f}".format(lpipss_t.mean()))

            scene_results[method] = {
                "SSIM": ssims_t.mean().item(),
                "PSNR": psnrs_t.mean().item(),
                "LPIPS": lpipss_t.mean().item(),
            }
            per_view_results[method] = {
                "SSIM": dict(zip(image_names, ssims_t.tolist())),
                "PSNR": dict(zip(image_names, psnrs_t.tolist())),
                "LPIPS": dict(zip(image_names, lpipss_t.tolist())),
            }

        with open(scene_path / f"results_{split}.json", "w") as fp:
            json.dump(scene_results, fp, indent=2)
        with open(scene_path / f"per_view_{split}.json", "w") as fp:
            json.dump(per_view_results, fp, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser(description="Streaming metric evaluation for rendered 3DGS images")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    torch.cuda.set_device(torch.device("cuda:0"))
    evaluate(args.model_paths, args.split)
