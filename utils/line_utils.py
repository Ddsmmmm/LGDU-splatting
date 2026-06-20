import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


def _starts_with(s, prefix):
    return s.startswith(prefix)


def _parse_obj_index(token):
    if "/" in token:
        token = token.split("/", 1)[0]
    return int(token)


def read_obj_segments(path):
    vertices = []
    segments = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if _starts_with(line, "v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif _starts_with(line, "l "):
                parts = line.split()[1:]
                idxs = [_parse_obj_index(tok) for tok in parts]
                for i in range(1, len(idxs)):
                    i0 = idxs[i - 1]
                    i1 = idxs[i]
                    if i0 <= 0 or i1 <= 0 or i0 > len(vertices) or i1 > len(vertices):
                        continue
                    segments.append((vertices[i0 - 1], vertices[i1 - 1]))
            elif _starts_with(line, "f "):
                parts = line.split()[1:]
                idxs = [_parse_obj_index(tok) for tok in parts]
                if len(idxs) >= 3:
                    for i in range(1, len(idxs)):
                        i0 = idxs[i - 1]
                        i1 = idxs[i]
                        if i0 <= 0 or i1 <= 0 or i0 > len(vertices) or i1 > len(vertices):
                            continue
                        segments.append((vertices[i0 - 1], vertices[i1 - 1]))
                    i0 = idxs[-1]
                    i1 = idxs[0]
                    if i0 > 0 and i1 > 0 and i0 <= len(vertices) and i1 <= len(vertices):
                        segments.append((vertices[i0 - 1], vertices[i1 - 1]))

    return segments


def read_limap_track_segments(path, min_visible_views=0, min_length=0.0):
    segments = []
    visible_counts = []
    support_counts = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return segments, visible_counts, support_counts

    n_tracks = int(lines[0].split()[0])
    cursor = 1
    for _ in range(n_tracks):
        header = lines[cursor].split()
        cursor += 1
        if len(header) < 3:
            break

        support_count = int(header[1])
        visible_count = int(header[2])
        start = [float(v) for v in lines[cursor].split()[:3]]
        cursor += 1
        end = [float(v) for v in lines[cursor].split()[:3]]
        cursor += 1

        # Skip image_id_list and line_id_list.
        cursor += 2

        length = math.sqrt(sum((start[i] - end[i]) ** 2 for i in range(3)))
        if visible_count < min_visible_views:
            continue
        if length < min_length:
            continue

        segments.append((start, end))
        visible_counts.append(visible_count)
        support_counts.append(support_count)

    return segments, visible_counts, support_counts


def _median(values, default=1.0):
    if not values:
        return default
    values = sorted(values)
    return values[len(values) // 2]


def _normalize_to_mean_one(weights, min_weight):
    if not weights:
        return weights
    mean = sum(weights) / max(len(weights), 1)
    if mean <= 0.0:
        return [1.0 for _ in weights]
    return [max(w / mean, min_weight) for w in weights]


def build_line_confidences(
    segments,
    visible_counts=None,
    support_counts=None,
    mode="length",
    power=1.0,
    min_weight=0.05,
):
    if mode == "none":
        return [1.0 for _ in segments]

    lengths = [
        math.sqrt(sum((seg[0][i] - seg[1][i]) ** 2 for i in range(3)))
        for seg in segments
    ]
    length_scale = max(_median(lengths), 1e-6)
    visible_scale = max(_median(visible_counts or []), 1.0)
    support_scale = max(_median(support_counts or []), 1.0)

    weights = []
    for idx, length in enumerate(lengths):
        weight = 1.0
        if "length" in mode:
            weight *= min(length / length_scale, 3.0)
        if visible_counts is not None and "visibility" in mode:
            weight *= min(math.log1p(visible_counts[idx]) / math.log1p(visible_scale), 3.0)
        if support_counts is not None and "support" in mode:
            weight *= min(math.log1p(support_counts[idx]) / math.log1p(support_scale), 3.0)
        weights.append(max(weight, min_weight) ** power)

    return _normalize_to_mean_one(weights, min_weight)


def load_line_segments(path, device="cuda"):
    segments = read_obj_segments(path)
    if not segments:
        raise RuntimeError("No line segments parsed from OBJ.")
    seg_tensor = torch.tensor(segments, dtype=torch.float32, device=device)
    return seg_tensor


def load_line_segments_with_confidence(
    obj_path="",
    tracks_path="",
    device="cuda",
    min_visible_views=0,
    min_length=0.0,
    confidence_mode="length",
    confidence_power=1.0,
    confidence_min=0.05,
):
    if tracks_path:
        segments, visible_counts, support_counts = read_limap_track_segments(
            tracks_path,
            min_visible_views=min_visible_views,
            min_length=min_length,
        )
    else:
        segments = read_obj_segments(obj_path)
        visible_counts = None
        support_counts = None
        if min_length > 0.0:
            segments = [
                seg for seg in segments
                if math.sqrt(sum((seg[0][i] - seg[1][i]) ** 2 for i in range(3))) >= min_length
            ]

    if not segments:
        raise RuntimeError("No line segments parsed after confidence filtering.")

    weights = build_line_confidences(
        segments,
        visible_counts=visible_counts,
        support_counts=support_counts,
        mode=confidence_mode,
        power=confidence_power,
        min_weight=confidence_min,
    )
    seg_tensor = torch.tensor(segments, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return seg_tensor, weight_tensor


def get_camera_edge_distance_map(camera, canny_low=50, canny_high=150):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for edge-supported line confidence.")

    key = (
        int(canny_low),
        int(canny_high),
        int(camera.image_width),
        int(camera.image_height),
    )
    cache = getattr(camera, "_line_edge_distance_cache", None)
    if cache is None:
        cache = {}
        setattr(camera, "_line_edge_distance_cache", cache)
    if key in cache:
        return cache[key]

    image = camera.original_image.detach().cpu().float().clamp(0.0, 1.0)
    image_np = image.permute(1, 2, 0).numpy()
    image_u8 = (image_np * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(canny_low), int(canny_high))

    # distanceTransform returns distance to zero pixels, so make edge pixels zero.
    non_edges = (edges == 0).astype(np.uint8)
    distances = cv2.distanceTransform(non_edges, cv2.DIST_L2, 3)
    distance_tensor = torch.from_numpy(distances.astype(np.float32))
    cache[key] = distance_tensor
    return distance_tensor


@torch.no_grad()
def projected_line_edge_support(
    segments,
    camera,
    canny_low=50,
    canny_high=150,
    sigma_px=2.0,
    sample_count=16,
    min_valid_ratio=0.5,
    min_projected_length=4.0,
    line_chunk_size=2048,
    return_valid=False,
):
    if segments.numel() == 0:
        support = torch.empty((0,), dtype=torch.float32, device=segments.device)
        if return_valid:
            return support, torch.empty((0,), dtype=torch.bool, device=segments.device)
        return support

    device = segments.device
    distance_map = get_camera_edge_distance_map(
        camera,
        canny_low=canny_low,
        canny_high=canny_high,
    ).to(device=device, non_blocking=True)
    height, width = distance_map.shape

    sample_count = max(int(sample_count), 2)
    line_chunk_size = max(int(line_chunk_size), 1)
    sigma_px = max(float(sigma_px), 1e-6)
    min_valid_ratio = max(float(min_valid_ratio), 0.0)
    min_projected_length = max(float(min_projected_length), 0.0)

    t = torch.linspace(0.0, 1.0, sample_count, device=device, dtype=segments.dtype)
    R = torch.as_tensor(camera.R, device=device, dtype=segments.dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=segments.dtype)
    focal_x = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    focal_y = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    znear = max(float(getattr(camera, "znear", 0.01)), 1e-4)

    supports = torch.zeros((segments.shape[0],), dtype=torch.float32, device=device)
    valid_flags = torch.zeros((segments.shape[0],), dtype=torch.bool, device=device)
    for start in range(0, segments.shape[0], line_chunk_size):
        end = min(start + line_chunk_size, segments.shape[0])
        seg = segments[start:end]
        a = seg[:, 0, :]
        b = seg[:, 1, :]
        points = a[:, None, :] * (1.0 - t[None, :, None]) + b[:, None, :] * t[None, :, None]
        flat_points = points.reshape(-1, 3)

        cam_points = flat_points @ R + T[None, :]
        z = cam_points[:, 2]
        z_safe = z.clamp_min(znear)
        u = cam_points[:, 0] / z_safe * focal_x + width * 0.5
        v = cam_points[:, 1] / z_safe * focal_y + height * 0.5

        u = u.reshape(-1, sample_count)
        v = v.reshape(-1, sample_count)
        z = z.reshape(-1, sample_count)

        valid = (
            (z > znear)
            & (u >= 0.0)
            & (u <= width - 1)
            & (v >= 0.0)
            & (v <= height - 1)
        )
        valid_count = valid.sum(dim=1)
        valid_ratio = valid_count.float() / float(sample_count)

        endpoint_valid = valid[:, 0] & valid[:, -1]
        projected_length = torch.sqrt(
            (u[:, -1] - u[:, 0]) ** 2 + (v[:, -1] - v[:, 0]) ** 2
        )
        line_valid = (
            (valid_ratio >= min_valid_ratio)
            & endpoint_valid
            & (projected_length >= min_projected_length)
        )

        uu = u.round().long().clamp(0, width - 1)
        vv = v.round().long().clamp(0, height - 1)
        distances = distance_map[vv, uu]
        sample_support = torch.exp(-0.5 * (distances / sigma_px) ** 2)
        sample_support = sample_support * valid.float()

        denom = valid_count.float().clamp_min(1.0)
        chunk_support = sample_support.sum(dim=1) / denom
        chunk_support = torch.where(line_valid, chunk_support, torch.zeros_like(chunk_support))
        supports[start:end] = chunk_support.float()
        valid_flags[start:end] = line_valid

    if return_valid:
        return supports, valid_flags
    return supports


@torch.no_grad()
def projected_line_mask(
    segments,
    camera,
    confidences=None,
    sample_count=32,
    dilation_px=2,
    min_valid_ratio=0.5,
    min_projected_length=4.0,
    line_chunk_size=2048,
    min_weight=0.0,
):
    if segments.numel() == 0:
        return torch.zeros((int(camera.image_height), int(camera.image_width)), dtype=torch.float32, device=segments.device)

    device = segments.device
    height = int(camera.image_height)
    width = int(camera.image_width)
    sample_count = max(int(sample_count), 2)
    line_chunk_size = max(int(line_chunk_size), 1)
    dilation_px = max(int(dilation_px), 0)
    min_valid_ratio = max(float(min_valid_ratio), 0.0)
    min_projected_length = max(float(min_projected_length), 0.0)
    min_weight = max(float(min_weight), 0.0)

    if confidences is None:
        confidences = torch.ones((segments.shape[0],), dtype=torch.float32, device=device)
    else:
        confidences = confidences.to(device=device, dtype=torch.float32)

    t = torch.linspace(0.0, 1.0, sample_count, device=device, dtype=segments.dtype)
    R = torch.as_tensor(camera.R, device=device, dtype=segments.dtype)
    T = torch.as_tensor(camera.T, device=device, dtype=segments.dtype)
    focal_x = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    focal_y = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    znear = max(float(getattr(camera, "znear", 0.01)), 1e-4)

    flat_mask = torch.zeros((height * width,), dtype=torch.float32, device=device)
    for start in range(0, segments.shape[0], line_chunk_size):
        end = min(start + line_chunk_size, segments.shape[0])
        seg = segments[start:end]
        weights = confidences[start:end].clamp_min(0.0)
        a = seg[:, 0, :]
        b = seg[:, 1, :]
        points = a[:, None, :] * (1.0 - t[None, :, None]) + b[:, None, :] * t[None, :, None]
        flat_points = points.reshape(-1, 3)

        cam_points = flat_points @ R + T[None, :]
        z = cam_points[:, 2]
        z_safe = z.clamp_min(znear)
        u = cam_points[:, 0] / z_safe * focal_x + width * 0.5
        v = cam_points[:, 1] / z_safe * focal_y + height * 0.5

        u = u.reshape(-1, sample_count)
        v = v.reshape(-1, sample_count)
        z = z.reshape(-1, sample_count)

        valid = (
            (z > znear)
            & (u >= 0.0)
            & (u <= width - 1)
            & (v >= 0.0)
            & (v <= height - 1)
        )
        valid_count = valid.sum(dim=1)
        valid_ratio = valid_count.float() / float(sample_count)
        endpoint_valid = valid[:, 0] & valid[:, -1]
        projected_length = torch.sqrt(
            (u[:, -1] - u[:, 0]) ** 2 + (v[:, -1] - v[:, 0]) ** 2
        )
        line_valid = (
            (valid_ratio >= min_valid_ratio)
            & endpoint_valid
            & (projected_length >= min_projected_length)
            & (weights > min_weight)
        )
        if not line_valid.any():
            continue

        uu = u.round().long().clamp(0, width - 1)
        vv = v.round().long().clamp(0, height - 1)
        sample_valid = valid & line_valid[:, None]
        indices = (vv * width + uu).reshape(-1)
        values = (weights[:, None].expand_as(valid).float() * sample_valid.float()).reshape(-1)
        sample_valid = sample_valid.reshape(-1)
        indices = indices[sample_valid]
        values = values[sample_valid]
        if indices.numel() > 0:
            flat_mask.index_put_((indices,), values, accumulate=True)

    mask = 1.0 - torch.exp(-flat_mask.clamp_min(0.0))
    mask = mask.reshape(1, 1, height, width)
    if dilation_px > 0:
        kernel_size = 2 * dilation_px + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=dilation_px)
    return mask.reshape(height, width).clamp(0.0, 1.0)


def point_to_segment_distance(points, segments, segment_chunk_size=256, point_chunk_size=4096, return_indices=False):
    if points.numel() == 0:
        distances = torch.empty((0,), device=points.device)
        if return_indices:
            return distances, torch.empty((0,), dtype=torch.long, device=points.device)
        return distances

    segment_chunk_size = max(int(segment_chunk_size), 1)
    point_chunk_size = max(int(point_chunk_size), 1)

    n = points.shape[0]
    m = segments.shape[0]
    mins_sq = torch.full((n,), float("inf"), device=points.device)
    min_indices = torch.zeros((n,), dtype=torch.long, device=points.device)
    eps = 1e-12

    for p_start in range(0, n, point_chunk_size):
        p_end = min(p_start + point_chunk_size, n)
        point_chunk = points[p_start:p_end]
        local_min_sq = torch.full((point_chunk.shape[0],), float("inf"), device=points.device)
        local_min_indices = torch.zeros((point_chunk.shape[0],), dtype=torch.long, device=points.device)

        for s_start in range(0, m, segment_chunk_size):
            s_end = min(s_start + segment_chunk_size, m)
            seg = segments[s_start:s_end]
            a = seg[:, 0, :]
            b = seg[:, 1, :]
            d = b - a
            len2 = (d * d).sum(dim=-1).clamp_min(eps)

            p_minus_a = point_chunk[:, None, :] - a[None, :, :]
            t = (p_minus_a * d[None, :, :]).sum(dim=-1) / len2[None, :]
            t = torch.clamp(t, 0.0, 1.0)
            proj = a[None, :, :] + d[None, :, :] * t[..., None]
            diff = point_chunk[:, None, :] - proj
            dist2 = (diff * diff).sum(dim=-1)
            chunk_min_sq, chunk_min_indices = dist2.min(dim=1)
            improved = chunk_min_sq < local_min_sq
            local_min_sq = torch.where(improved, chunk_min_sq, local_min_sq)
            local_min_indices = torch.where(improved, chunk_min_indices + s_start, local_min_indices)

        mins_sq[p_start:p_end] = local_min_sq
        min_indices[p_start:p_end] = local_min_indices

    distances = torch.sqrt(mins_sq)
    if return_indices:
        return distances, min_indices
    return distances
