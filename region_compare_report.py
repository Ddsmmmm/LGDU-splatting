"""Create a compact report from region_metrics.py outputs."""

from argparse import ArgumentParser
from pathlib import Path
import json


HIGHER_IS_BETTER = {"PSNR", "SSIM"}
LOWER_IS_BETTER = {"LPIPS"}
METRICS = ("PSNR", "SSIM", "LPIPS")


def load_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def metric_mean(results, method, region, metric):
    item = results.get(method, {}).get(region, {}).get(metric)
    if item is None:
        return None
    return item.get("mean")


def metric_count(results, method, region, metric):
    item = results.get(method, {}).get(region, {}).get(metric)
    if item is None:
        return 0
    return int(item.get("count", 0))


def metric_delta(results, baseline, method, region, metric):
    base_value = metric_mean(results, baseline, region, metric)
    method_value = metric_mean(results, method, region, metric)
    if base_value is None or method_value is None:
        return None
    return method_value - base_value


def is_improved(delta, metric):
    if delta is None:
        return False
    if metric in HIGHER_IS_BETTER:
        return delta > 0
    if metric in LOWER_IS_BETTER:
        return delta < 0
    return False


def signed_fmt(delta, metric, digits=4):
    if delta is None:
        return "-"
    sign = "+" if delta >= 0 else ""
    arrow = "better" if is_improved(delta, metric) else "worse"
    return f"{sign}{delta:.{digits}f} ({arrow})"


def win_rate(per_view, baseline, method, region, metric):
    baseline_views = per_view.get(baseline, {}).get(region, {})
    method_views = per_view.get(method, {}).get(region, {})
    names = sorted(set(baseline_views.keys()) & set(method_views.keys()))
    if not names:
        return None

    wins = 0
    for name in names:
        base_value = baseline_views[name].get(metric)
        method_value = method_views[name].get(metric)
        if base_value is None or method_value is None:
            continue
        if metric in HIGHER_IS_BETTER and method_value > base_value:
            wins += 1
        elif metric in LOWER_IS_BETTER and method_value < base_value:
            wins += 1
    return wins, len(names), wins / max(len(names), 1)


def region_names(results):
    regions = set()
    for method_results in results.values():
        regions.update(method_results.keys())
    return sorted(regions)


def aggregate_table(results, baseline, methods, regions):
    lines = []
    lines.append("| Region | Method | Count | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Mask |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for region in regions:
        for method in methods:
            count = metric_count(results, method, region, "PSNR")
            psnr = metric_mean(results, method, region, "PSNR")
            ssim = metric_mean(results, method, region, "SSIM")
            lpips = metric_mean(results, method, region, "LPIPS")
            mask = metric_mean(results, method, region, "mask_ratio")
            if method == baseline:
                d_psnr = d_ssim = d_lpips = None
            else:
                d_psnr = metric_delta(results, baseline, method, region, "PSNR")
                d_ssim = metric_delta(results, baseline, method, region, "SSIM")
                d_lpips = metric_delta(results, baseline, method, region, "LPIPS")
            lines.append(
                "| {region} | {method} | {count} | {psnr} | {d_psnr} | {ssim} | {d_ssim} | {lpips} | {d_lpips} | {mask} |".format(
                    region=region,
                    method=method,
                    count=count,
                    psnr=fmt(psnr),
                    d_psnr=signed_fmt(d_psnr, "PSNR"),
                    ssim=fmt(ssim, 5),
                    d_ssim=signed_fmt(d_ssim, "SSIM", 5),
                    lpips=fmt(lpips, 5),
                    d_lpips=signed_fmt(d_lpips, "LPIPS", 5),
                    mask=fmt(mask, 5),
                )
            )
    return "\n".join(lines)


def win_rate_table(per_view, baseline, methods, regions):
    lines = []
    lines.append("| Region | Method | PSNR win | SSIM win | LPIPS win |")
    lines.append("|---|---:|---:|---:|---:|")
    for region in regions:
        for method in methods:
            if method == baseline:
                continue
            values = {}
            for metric in METRICS:
                rate = win_rate(per_view, baseline, method, region, metric)
                if rate is None:
                    values[metric] = "-"
                else:
                    wins, total, frac = rate
                    values[metric] = f"{wins}/{total} ({frac * 100:.1f}%)"
            lines.append(
                f"| {region} | {method} | {values['PSNR']} | {values['SSIM']} | {values['LPIPS']} |"
            )
    return "\n".join(lines)


def automatic_findings(results, per_view, baseline, methods, regions):
    lines = []
    for region in regions:
        for method in methods:
            if method == baseline:
                continue
            improved = []
            worsened = []
            for metric in METRICS:
                delta = metric_delta(results, baseline, method, region, metric)
                if is_improved(delta, metric):
                    improved.append(metric)
                else:
                    worsened.append(metric)
            wins = []
            for metric in METRICS:
                rate = win_rate(per_view, baseline, method, region, metric)
                if rate is not None and rate[2] > 0.5:
                    wins.append(metric)
            if improved:
                lines.append(
                    f"- `{method}` improves `{region}` over `{baseline}` on aggregate {', '.join(improved)}; per-view majority wins: {', '.join(wins) if wins else 'none'}."
                )
            else:
                lines.append(
                    f"- `{method}` does not improve `{region}` over `{baseline}` on aggregate metrics."
                )
    return "\n".join(lines)


def write_report(args):
    results = load_json(args.results)
    per_view = load_json(args.per_view)

    methods = args.methods if args.methods else list(results.keys())
    missing = [method for method in methods if method not in results]
    if missing:
        raise ValueError(f"Missing methods in results JSON: {missing}")
    if args.baseline not in results:
        raise ValueError(f"Baseline not found in results JSON: {args.baseline}")

    regions = args.regions if args.regions else region_names(results)
    lines = [
        "# Region Comparison Report",
        "",
        f"Baseline: `{args.baseline}`",
        "",
        "## Aggregate Metrics",
        "",
        aggregate_table(results, args.baseline, methods, regions),
        "",
        "## Per-View Win Rate vs Baseline",
        "",
        win_rate_table(per_view, args.baseline, methods, regions),
        "",
        "## Auto Findings",
        "",
        automatic_findings(results, per_view, args.baseline, methods, regions),
        "",
    ]
    if args.crop_dir:
        lines.extend(["## Crop Directory", "", f"`{args.crop_dir}`", ""])

    text = "\n".join(lines)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Report written to: {output}")
    else:
        print(text)


if __name__ == "__main__":
    parser = ArgumentParser(description="Summarize local region metrics against a baseline.")
    parser.add_argument("--results", required=True, type=str)
    parser.add_argument("--per_view", required=True, type=str)
    parser.add_argument("--baseline", required=True, type=str)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--crop_dir", default="", type=str)
    parser.add_argument("--output", default="", type=str)
    write_report(parser.parse_args())
