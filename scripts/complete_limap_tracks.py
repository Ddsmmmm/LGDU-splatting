#!/usr/bin/env python3
"""Conservative topology completion for LIMAP alltracks.txt files.

The script keeps the original LIMAP tracks unchanged and appends synthetic
tracks for two safe cases:

1. parallel/collinear fragments with a small endpoint gap;
2. near-orthogonal endpoints whose infinite 3D lines meet near both endpoints.

The output remains compatible with LGDU's --line_tracks_path loader.
"""

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Track:
    track_id: int
    support_count: int
    visible_count: int
    start: np.ndarray
    end: np.ndarray
    image_ids: list
    line_ids: list

    @property
    def direction(self):
        vec = self.end - self.start
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return None
        return vec / norm

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))


@dataclass
class SyntheticTrack:
    kind: str
    parent_a: int
    parent_b: int
    start: np.ndarray
    end: np.ndarray
    image_ids: list
    support_count: int
    visible_count: int

    @property
    def length(self):
        return float(np.linalg.norm(self.end - self.start))


def parse_alltracks(path):
    raw_lines = [line.rstrip("\n") for line in Path(path).read_text().splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"Empty alltracks file: {path}")

    n_tracks = int(raw_lines[0].split()[0])
    tracks = []
    cursor = 1
    for _ in range(n_tracks):
        header = raw_lines[cursor].split()
        cursor += 1
        if len(header) < 3:
            raise ValueError(f"Malformed track header near line {cursor}: {header}")

        track_id = int(header[0])
        support_count = int(header[1])
        visible_count = int(header[2])
        start = np.array([float(v) for v in raw_lines[cursor].split()[:3]], dtype=np.float64)
        cursor += 1
        end = np.array([float(v) for v in raw_lines[cursor].split()[:3]], dtype=np.float64)
        cursor += 1
        image_ids = [int(v) for v in raw_lines[cursor].split()]
        cursor += 1
        line_ids = [int(v) for v in raw_lines[cursor].split()]
        cursor += 1
        tracks.append(
            Track(
                track_id=track_id,
                support_count=support_count,
                visible_count=visible_count,
                start=start,
                end=end,
                image_ids=image_ids,
                line_ids=line_ids,
            )
        )

    if cursor != len(raw_lines):
        print(f"[warn] Parsed {cursor} non-empty lines, file contains {len(raw_lines)} lines.")

    return tracks


def write_alltracks(path, tracks, synthetic_tracks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    next_id = max((track.track_id for track in tracks), default=-1) + 1

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(tracks) + len(synthetic_tracks)}\n")
        for track in tracks:
            f.write(f"{track.track_id} {track.support_count} {track.visible_count}\n")
            f.write(format_point(track.start))
            f.write(format_point(track.end))
            f.write(format_int_list(track.image_ids))
            f.write(format_int_list(track.line_ids))

        for idx, track in enumerate(synthetic_tracks):
            support_count = max(1, int(track.support_count))
            visible_count = max(1, int(track.visible_count))
            image_ids = list(track.image_ids)
            if not image_ids:
                image_ids = [0] * visible_count
            if len(image_ids) < visible_count:
                image_ids = image_ids + [image_ids[-1]] * (visible_count - len(image_ids))
            image_ids = image_ids[:visible_count]
            line_ids = [-1] * support_count

            f.write(f"{next_id + idx} {support_count} {visible_count}\n")
            f.write(format_point(track.start))
            f.write(format_point(track.end))
            f.write(format_int_list(image_ids))
            f.write(format_int_list(line_ids))


def write_obj(path, tracks, synthetic_tracks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Original tracks followed by completed synthetic tracks.\n")
        vertex_id = 1
        for track in tracks:
            f.write(f"v {track.start[0]:.10f} {track.start[1]:.10f} {track.start[2]:.10f}\n")
            f.write(f"v {track.end[0]:.10f} {track.end[1]:.10f} {track.end[2]:.10f}\n")
            f.write(f"l {vertex_id} {vertex_id + 1}\n")
            vertex_id += 2
        f.write("# Synthetic completion tracks.\n")
        for track in synthetic_tracks:
            f.write(f"v {track.start[0]:.10f} {track.start[1]:.10f} {track.start[2]:.10f}\n")
            f.write(f"v {track.end[0]:.10f} {track.end[1]:.10f} {track.end[2]:.10f}\n")
            f.write(f"l {vertex_id} {vertex_id + 1}\n")
            vertex_id += 2


def format_point(point):
    return f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f}\n"


def format_int_list(values):
    return " ".join(str(int(v)) for v in values) + " \n"


def angle_degrees(u, v):
    dot = float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))
    return math.degrees(math.acos(dot))


def closest_points_on_lines(p1, u, p2, v):
    # Solve p1 + s u ~= p2 + t v for closest infinite-line points.
    w0 = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    q1 = p1 + s * u
    q2 = p2 + t * v
    return q1, q2, s, t


def segment_parameter(point, track):
    vec = track.end - track.start
    denom = float(np.dot(vec, vec))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(point - track.start, vec) / denom)


def nearest_endpoint(point, track):
    d_start = float(np.linalg.norm(point - track.start))
    d_end = float(np.linalg.norm(point - track.end))
    if d_start <= d_end:
        return track.start, d_start, "start"
    return track.end, d_end, "end"


def support_metadata(track_a, track_b, support_scale, min_visible_views):
    intersection = sorted(set(track_a.image_ids).intersection(track_b.image_ids))
    if len(intersection) >= min_visible_views:
        image_ids = intersection
    else:
        shorter = track_a.image_ids if len(track_a.image_ids) <= len(track_b.image_ids) else track_b.image_ids
        image_ids = list(shorter)

    visible_count = min(track_a.visible_count, track_b.visible_count, len(image_ids))
    visible_count = max(visible_count, min_visible_views)
    support_count = round(min(track_a.support_count, track_b.support_count) * support_scale)
    support_count = max(min_visible_views, support_count)
    return image_ids, support_count, visible_count


def build_endpoint_index(tracks, cell_size):
    grid = defaultdict(list)
    endpoints = []
    for track_idx, track in enumerate(tracks):
        for endpoint_idx, point in enumerate((track.start, track.end)):
            key = tuple(np.floor(point / cell_size).astype(np.int64).tolist())
            grid[key].append(len(endpoints))
            endpoints.append((track_idx, endpoint_idx, point))
    return grid, endpoints


def iter_neighbor_endpoint_pairs(tracks, max_gap):
    if max_gap <= 0.0:
        return
    cell_size = max(max_gap, 1e-8)
    grid, endpoints = build_endpoint_index(tracks, cell_size)
    seen = set()
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for idx, (track_idx, endpoint_idx, point) in enumerate(endpoints):
        key = tuple(np.floor(point / cell_size).astype(np.int64).tolist())
        for offset in offsets:
            neighbor_key = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
            for other_idx in grid.get(neighbor_key, []):
                if other_idx <= idx:
                    continue
                other_track_idx, other_endpoint_idx, other_point = endpoints[other_idx]
                if track_idx == other_track_idx:
                    continue
                distance = float(np.linalg.norm(point - other_point))
                if distance > max_gap:
                    continue
                pair_key = (min(idx, other_idx), max(idx, other_idx))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                yield track_idx, endpoint_idx, other_track_idx, other_endpoint_idx, distance


def add_unique_segment(segments, candidate, duplicate_tol):
    a = candidate.start
    b = candidate.end
    if np.linalg.norm(a - b) < 1e-12:
        return False
    key_a = tuple(np.round(a / duplicate_tol).astype(np.int64).tolist())
    key_b = tuple(np.round(b / duplicate_tol).astype(np.int64).tolist())
    key = tuple(sorted((key_a, key_b)))
    if key in segments:
        return False
    segments.add(key)
    return True


def complete_tracks(tracks, args):
    reliable = [
        track
        for track in tracks
        if track.length >= args.min_track_length
        and track.visible_count >= args.min_visible_views
        and track.direction is not None
    ]
    if not reliable:
        return [], {"reliable": 0, "parallel": 0, "corner": 0, "candidates": 0}

    lengths = np.array([track.length for track in reliable], dtype=np.float64)
    median_length = float(np.median(lengths))
    max_gap = args.max_endpoint_gap
    if max_gap <= 0.0:
        max_gap = median_length * args.max_endpoint_gap_ratio
    max_intersection_distance = args.max_intersection_distance
    if max_intersection_distance <= 0.0:
        max_intersection_distance = max_gap * args.max_intersection_distance_ratio

    synthetic = []
    occupied = set()
    duplicate_tol = max(args.duplicate_tolerance, max_gap * 0.05, 1e-6)
    for track in tracks:
        add_unique_segment(
            occupied,
            SyntheticTrack("original", track.track_id, track.track_id, track.start, track.end, [], 1, 1),
            duplicate_tol,
        )

    stats = {"reliable": len(reliable), "parallel": 0, "corner": 0, "candidates": 0}
    for idx_a, _, idx_b, _, endpoint_distance in iter_neighbor_endpoint_pairs(reliable, max_gap):
        track_a = reliable[idx_a]
        track_b = reliable[idx_b]
        u = track_a.direction
        v = track_b.direction
        angle = angle_degrees(u, v)
        stats["candidates"] += 1

        if angle <= args.parallel_angle_deg:
            created = make_parallel_bridge(
                track_a,
                track_b,
                endpoint_distance,
                max_gap,
                max_intersection_distance,
                args,
            )
        elif abs(angle - 90.0) <= args.corner_angle_tolerance_deg:
            created = make_corner_extensions(
                track_a,
                track_b,
                max_gap,
                max_intersection_distance,
                args,
            )
        else:
            created = []

        for candidate in created:
            if candidate.length < args.min_completion_length:
                continue
            if candidate.length > max_gap * args.max_completion_gap_factor:
                continue
            if add_unique_segment(occupied, candidate, duplicate_tol):
                synthetic.append(candidate)
                stats[candidate.kind] += 1

    stats["pre_cap_total"] = len(synthetic)
    if args.max_new_tracks > 0 and len(synthetic) > args.max_new_tracks:
        synthetic = sorted(synthetic, key=lambda item: item.length)[: args.max_new_tracks]
        stats["parallel"] = sum(1 for item in synthetic if item.kind == "parallel")
        stats["corner"] = sum(1 for item in synthetic if item.kind == "corner")

    return synthetic, stats


def make_parallel_bridge(track_a, track_b, endpoint_distance, max_gap, max_intersection_distance, args):
    u = track_a.direction
    midpoint_b = 0.5 * (track_b.start + track_b.end)
    line_distance = float(np.linalg.norm(np.cross(midpoint_b - track_a.start, u)))
    if line_distance > max_intersection_distance:
        return []

    endpoints = [
        (track_a.start, track_b.start),
        (track_a.start, track_b.end),
        (track_a.end, track_b.start),
        (track_a.end, track_b.end),
    ]
    p, q = min(endpoints, key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
    if float(np.linalg.norm(p - q)) > max_gap:
        return []

    image_ids, support_count, visible_count = support_metadata(
        track_a, track_b, args.synthetic_support_scale, args.min_visible_views
    )
    return [
        SyntheticTrack(
            kind="parallel",
            parent_a=track_a.track_id,
            parent_b=track_b.track_id,
            start=p,
            end=q,
            image_ids=image_ids,
            support_count=support_count,
            visible_count=visible_count,
        )
    ]


def make_corner_extensions(track_a, track_b, max_gap, max_intersection_distance, args):
    result = closest_points_on_lines(track_a.start, track_a.direction, track_b.start, track_b.direction)
    if result is None:
        return []
    q1, q2, _, _ = result
    if float(np.linalg.norm(q1 - q2)) > max_intersection_distance:
        return []
    intersection = 0.5 * (q1 + q2)

    endpoint_a, dist_a, _ = nearest_endpoint(intersection, track_a)
    endpoint_b, dist_b, _ = nearest_endpoint(intersection, track_b)
    if dist_a > max_gap or dist_b > max_gap:
        return []

    t_a = segment_parameter(intersection, track_a)
    t_b = segment_parameter(intersection, track_b)
    # For corner completion, the intersection should be near or just outside an
    # endpoint, not deep in the middle of both existing segments.
    if args.require_outside_endpoint and (0.05 < t_a < 0.95) and (0.05 < t_b < 0.95):
        return []

    max_extension_a = max(track_a.length * args.max_extension_ratio, args.min_completion_length)
    max_extension_b = max(track_b.length * args.max_extension_ratio, args.min_completion_length)
    if dist_a > max_extension_a or dist_b > max_extension_b:
        return []

    image_ids, support_count, visible_count = support_metadata(
        track_a, track_b, args.synthetic_support_scale, args.min_visible_views
    )
    candidates = []
    if dist_a >= args.min_completion_length:
        candidates.append(
            SyntheticTrack(
                kind="corner",
                parent_a=track_a.track_id,
                parent_b=track_b.track_id,
                start=endpoint_a,
                end=intersection,
                image_ids=image_ids,
                support_count=support_count,
                visible_count=visible_count,
            )
        )
    if dist_b >= args.min_completion_length:
        candidates.append(
            SyntheticTrack(
                kind="corner",
                parent_a=track_a.track_id,
                parent_b=track_b.track_id,
                start=endpoint_b,
                end=intersection,
                image_ids=image_ids,
                support_count=support_count,
                visible_count=visible_count,
            )
        )
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append conservative completed tracks to a LIMAP alltracks.txt file."
    )
    parser.add_argument("--input", required=True, help="Input LIMAP alltracks.txt")
    parser.add_argument("--output", required=True, help="Output completed alltracks.txt")
    parser.add_argument("--output_obj", default="", help="Optional OBJ for visual inspection")
    parser.add_argument("--min_visible_views", type=int, default=6)
    parser.add_argument("--min_track_length", type=float, default=0.0)
    parser.add_argument("--max_endpoint_gap", type=float, default=0.0, help="Absolute 3D gap. <=0 uses median length ratio.")
    parser.add_argument("--max_endpoint_gap_ratio", type=float, default=0.35)
    parser.add_argument("--max_intersection_distance", type=float, default=0.0)
    parser.add_argument("--max_intersection_distance_ratio", type=float, default=0.25)
    parser.add_argument("--parallel_angle_deg", type=float, default=8.0)
    parser.add_argument("--corner_angle_tolerance_deg", type=float, default=15.0)
    parser.add_argument("--max_extension_ratio", type=float, default=0.75)
    parser.add_argument("--max_completion_gap_factor", type=float, default=1.5)
    parser.add_argument("--min_completion_length", type=float, default=1e-4)
    parser.add_argument("--synthetic_support_scale", type=float, default=0.5)
    parser.add_argument("--duplicate_tolerance", type=float, default=1e-5)
    parser.add_argument("--max_new_tracks", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--allow_mid_segment_intersections", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.require_outside_endpoint = not args.allow_mid_segment_intersections
    tracks = parse_alltracks(args.input)
    synthetic, stats = complete_tracks(tracks, args)
    write_alltracks(args.output, tracks, synthetic)
    if args.output_obj:
        write_obj(args.output_obj, tracks, synthetic)

    print(f"Input tracks: {len(tracks)}")
    print(f"Reliable tracks used: {stats['reliable']}")
    print(f"Endpoint candidate pairs: {stats['candidates']}")
    if stats.get("pre_cap_total", len(synthetic)) != len(synthetic):
        print(f"Synthetic tracks before cap: {stats['pre_cap_total']}")
    print(f"Added synthetic tracks: {len(synthetic)}")
    print(f"  parallel bridges: {stats['parallel']}")
    print(f"  corner extensions: {stats['corner']}")
    print(f"Output alltracks: {args.output}")
    if args.output_obj:
        print(f"Output OBJ: {args.output_obj}")


if __name__ == "__main__":
    main()
