"""
Evaluation harness for the label placement problem.

Scores label placements on:
  1. Line overlap — labels intersecting connection lines between consecutive points
  2. Point overlap — labels intersecting their own data point (radius 8px)
  3. Label overlap — labels intersecting each other
  4. Canvas clipping — label boxes extending beyond canvas edges
  5. Proximity — average distance from label centers to data points
  6. Stability — how much label positions change when canvas size changes

Run: python eval.py
Output: JSON with score, description, and per-metric breakdown.
"""

import json
import math
import random
import sys
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
NUM_CONFIGS = 5
NUM_POINTS = 10
LABEL_W = 100
LABEL_H = 40
POINT_RADIUS = 8

# Canvas scenarios: (canvas_w, canvas_h)
BASELINE_SCENARIO = (800, 600)

STABILITY_SCENARIOS = [
    (600, 450),
    (1000, 750),
    (400, 300),
    (1200, 900),
    (1400, 1000),
]

# Margins in normalized [0,1] space for data generation
X_MARGIN = 0.05   # 5% from left/right edges
Y_MARGIN = 0.034  # ~3.4% from top/bottom edges


# ---------------------------------------------------------------------------
# Data generation (normalized coordinates in [0,1])
# ---------------------------------------------------------------------------

def generate_config(rng, num_points=NUM_POINTS):
    """Generate a single test configuration with monotonic y values.

    Returns normalized coordinates in [0, 1]. These are scaled to
    pixel coordinates at evaluation time based on canvas size.
    """
    x_lo, x_hi = X_MARGIN, 1.0 - X_MARGIN
    y_lo, y_hi = Y_MARGIN, 1.0 - Y_MARGIN

    # Random x positions, sorted
    xs = sorted(rng.uniform(x_lo, x_hi) for _ in range(num_points))
    # Ensure minimum spacing between consecutive x values
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] < 0.04:
            xs[i] = xs[i - 1] + 0.04

    # Random monotonic y direction
    increasing = rng.choice([True, False])
    ys_base = sorted(rng.uniform(y_lo, y_hi) for _ in range(num_points))
    if not increasing:
        ys_base = list(reversed(ys_base))

    return xs, ys_base


def generate_all_configs():
    """Generate NUM_CONFIGS random test configurations."""
    rng = random.Random(SEED)
    configs = []
    for _ in range(NUM_CONFIGS):
        xs, ys = generate_config(rng)
        configs.append((xs, ys))
    return configs


def rescale(normalized_x, normalized_y, canvas_w, canvas_h):
    """Scale normalized [0,1] coordinates to pixel coordinates for a canvas."""
    px = [x * canvas_w for x in normalized_x]
    py = [y * canvas_h for y in normalized_y]
    return px, py


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rect_overlap_area(x1, y1, w1, h1, x2, y2, w2, h2):
    """Return overlapping area between two axis-aligned rectangles given center coords."""
    left1, right1 = x1 - w1 / 2, x1 + w1 / 2
    bottom1, top1 = y1 - h1 / 2, y1 + h1 / 2
    left2, right2 = x2 - w2 / 2, x2 + w2 / 2
    bottom2, top2 = y2 - h2 / 2, y2 + h2 / 2

    overlap_x = max(0, min(right1, right2) - max(left1, left2))
    overlap_y = max(0, min(top1, top2) - max(bottom1, bottom2))
    return overlap_x * overlap_y


def point_in_rect(px, py, cx, cy, w, h):
    """Check if a point is inside a rectangle (given center coords)."""
    return (cx - w / 2 <= px <= cx + w / 2 and
            cy - h / 2 <= py <= cy + h / 2)


def point_in_rect_expanded(px, py, cx, cy, w, h, radius):
    """Check if a point is inside a rectangle expanded by a radius."""
    return (cx - w / 2 - radius <= px <= cx + w / 2 + radius and
            cy - h / 2 - radius <= py <= cy + h / 2 + radius)


def segment_rect_intersect(px1, py1, px2, py2, cx, cy, w, h):
    """Check if a line segment intersects a rectangle."""
    if point_in_rect(px1, py1, cx, cy, w, h):
        return True
    if point_in_rect(px2, py2, cx, cy, w, h):
        return True

    rect_edges = [
        (cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2),  # top
        (cx - w / 2, cy + h / 2, cx + w / 2, cy + h / 2),  # bottom
        (cx - w / 2, cy - h / 2, cx - w / 2, cy + h / 2),  # left
        (cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2),  # right
    ]

    for x1, y1, x2, y2 in rect_edges:
        if _segments_intersect(px1, py1, px2, py2, x1, y1, x2, y2):
            return True
    return False


def _segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """Check if two line segments intersect."""
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)

    d1 = ccw(x3, y3, x4, y4, x1, y1)
    d2 = ccw(x3, y3, x4, y4, x2, y2)
    d3 = ccw(x1, y1, x2, y2, x3, y3)
    d4 = ccw(x1, y1, x2, y2, x4, y4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and _on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if d2 == 0 and _on_segment(x3, y3, x4, y4, x2, y2):
        return True
    if d3 == 0 and _on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if d4 == 0 and _on_segment(x1, y1, x2, y2, x4, y4):
        return True

    return False


def _on_segment(x1, y1, x2, y2, px, py):
    """Check if point (px, py) lies on segment (x1,y1)-(x2,y2)."""
    return (min(x1, x2) <= px <= max(x1, x2) and
            min(y1, y2) <= py <= max(y1, y2))


def stepped_segment_rect_intersect(px1, py1, px2, py2, cx, cy, w, h):
    """Check if a stepped line ('before' style) intersects a rectangle.

    The path goes horizontally from (px1,py1) to (px2,py1),
    then vertically from (px2,py1) to (px2,py2).
    """
    if segment_rect_intersect(px1, py1, px2, py1, cx, cy, w, h):
        return True
    if segment_rect_intersect(px2, py1, px2, py2, cx, cy, w, h):
        return True
    return False


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def get_per_label_line_overlaps(labels, points_x, points_y):
    """Per-label count of intersected line segments.

    Returns a list of dicts: [{label: i, segments_hit: [...], count: N}, ...]
    """
    n = len(labels)
    results = []
    for i in range(n):
        lx, ly = labels[i]
        hit_segments = []
        for j in range(len(points_x) - 1):
            if stepped_segment_rect_intersect(
                points_x[j], points_y[j],
                points_x[j + 1], points_y[j + 1],
                lx, ly, LABEL_W, LABEL_H,
            ):
                hit_segments.append(j)
        results.append({"label": i, "segments_hit": hit_segments, "count": len(hit_segments)})
    return results


def score_line_overlaps(labels, points_x, points_y):
    """Score for label boxes intersecting connection lines."""
    n = len(labels)
    max_segments = max(n - 1, 1) * n
    total = 0.0
    for i in range(n):
        lx, ly = labels[i]
        for j in range(len(points_x) - 1):
            if stepped_segment_rect_intersect(
                points_x[j], points_y[j],
                points_x[j + 1], points_y[j + 1],
                lx, ly, LABEL_W, LABEL_H,
            ):
                total += 1.0
    return total / max_segments


def get_per_label_point_overlaps(labels, points_x, points_y):
    """Per-label point overlap info.

    Returns a list of dicts: [{label: i, overlaps_point: bool, dist_to_point: float}, ...]
    """
    n = len(labels)
    results = []
    for i in range(n):
        lx, ly = labels[i]
        overlaps = point_in_rect_expanded(points_x[i], points_y[i], lx, ly, LABEL_W, LABEL_H, POINT_RADIUS)
        dx = lx - points_x[i]
        dy = ly - points_y[i]
        dist = math.sqrt(dx * dx + dy * dy)
        results.append({"label": i, "overlaps_point": overlaps, "dist_to_point": round(dist, 1)})
    return results


def score_point_overlaps(labels, points_x, points_y):
    """Score for label boxes overlapping their own data points."""
    n = len(labels)
    total = 0.0
    for i in range(n):
        lx, ly = labels[i]
        if point_in_rect_expanded(points_x[i], points_y[i], lx, ly, LABEL_W, LABEL_H, POINT_RADIUS):
            total += 1.0
    return total / n


def get_per_label_label_overlaps(labels):
    """Per-label pair overlaps.

    Returns a list of dicts: [{label_i: i, label_j: j, overlap_frac: float}, ...]
    """
    n = len(labels)
    results = []
    for i, j in combinations(range(n), 2):
        lx1, ly1 = labels[i]
        lx2, ly2 = labels[j]
        overlap = rect_overlap_area(lx1, ly1, LABEL_W, LABEL_H,
                                             lx2, ly2, LABEL_W, LABEL_H)
        frac = overlap / (LABEL_W * LABEL_H)
        if frac > 0:
            results.append({"label_i": i, "label_j": j, "overlap_frac": round(frac, 4)})
    return results


def score_label_overlaps(labels):
    """Score for label boxes overlapping each other."""
    n = len(labels)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        lx1, ly1 = labels[i]
        lx2, ly2 = labels[j]
        overlap = rect_overlap_area(lx1, ly1, LABEL_W, LABEL_H,
                                             lx2, ly2, LABEL_W, LABEL_H)
        total += overlap / (LABEL_W * LABEL_H)
        count += 1
    return total / count


def get_per_label_clipping(labels, canvas_w, canvas_h):
    """Per-label clipping info.

    Returns a list of dicts: [{label: i, visible_frac: float, clipped: bool,
    left: float, right: float, top: float, bottom: float}, ...]
    """
    n = len(labels)
    results = []
    for i in range(n):
        lx, ly = labels[i]
        box_left = lx - LABEL_W / 2
        box_right = lx + LABEL_W / 2
        box_top = ly - LABEL_H / 2
        box_bottom = ly + LABEL_H / 2

        vis_left = max(0, box_left)
        vis_right = min(canvas_w, box_right)
        vis_bottom = max(0, box_top)
        vis_top = min(canvas_h, box_bottom)

        visible_w = max(0, vis_right - vis_left)
        visible_h = max(0, vis_top - vis_bottom)
        visible_area = visible_w * visible_h
        visible_frac = visible_area / (LABEL_W * LABEL_H)
        results.append({
            "label": i,
            "visible_frac": round(visible_frac, 4),
            "clipped": round(visible_frac, 4) < 1.0,
            "box_left": round(box_left, 1), "box_right": round(box_right, 1),
            "box_top": round(box_top, 1), "box_bottom": round(box_bottom, 1),
        })
    return results


def score_clipping(labels, canvas_w, canvas_h):
    """Score for label boxes clipped by canvas edges."""
    n = len(labels)
    total = 0.0
    for lx, ly in labels:
        left = max(0, lx - LABEL_W / 2)
        right = min(canvas_w, lx + LABEL_W / 2)
        bottom = max(0, ly - LABEL_H / 2)
        top = min(canvas_h, ly + LABEL_H / 2)

        visible_w = max(0, right - left)
        visible_h = max(0, top - bottom)
        visible_area = visible_w * visible_h
        total += 1.0 - (visible_area / (LABEL_W * LABEL_H))
    return total / n


def get_per_label_proximity(labels, points_x, points_y, canvas_w, canvas_h):
    """Per-label proximity info.

    Returns a list of dicts: [{label: i, dist_px: float, dist_frac: float, score: float}, ...]
    """
    n = len(labels)
    diag = math.sqrt(canvas_w ** 2 + canvas_h ** 2)
    results = []
    for i in range(n):
        dx = labels[i][0] - points_x[i]
        dy = labels[i][1] - points_y[i]
        dist_px = math.sqrt(dx * dx + dy * dy)
        dist_frac = dist_px / diag
        score = 1.0 - math.tanh(dist_frac / 0.05)
        results.append({"label": i, "dist_px": round(dist_px, 1), "dist_frac": round(dist_frac, 4), "score": round(score, 4)})
    return results


def score_proximity(labels, points_x, points_y, canvas_w, canvas_h):
    """Score for average distance from label centers to data points.

    Distance is normalized by canvas diagonal so the threshold scales with
    canvas size. Returns score in [0, 1] where 1 = perfect (on point).
    """
    n = len(labels)
    diag = math.sqrt(canvas_w ** 2 + canvas_h ** 2)
    total_dist = 0.0
    for i in range(n):
        dx = labels[i][0] - points_x[i]
        dy = labels[i][1] - points_y[i]
        total_dist += math.sqrt(dx * dx + dy * dy) / diag
    avg_dist = total_dist / n
    return 1.0 - math.tanh(avg_dist / 0.05)


def score_stability(place_labels_fn, norm_px, norm_py, canvas_w, canvas_h):
    """Score for stability of label positions across canvas sizes.

    For each alternate canvas size, rescales the data points proportionally
    and checks that label positions follow the same scaling.
    Returns score in [0, 1] where 1 = perfectly stable.
    """
    baseline_px, baseline_py = rescale(norm_px, norm_py,
                                       BASELINE_SCENARIO[0], BASELINE_SCENARIO[1])
    baseline_labels = place_labels_fn(*BASELINE_SCENARIO, baseline_px, baseline_py, LABEL_W, LABEL_H)

    if len(baseline_labels) != len(norm_px):
        return 0.0

    total_disp = 0.0
    count = 0
    for cw, ch in STABILITY_SCENARIOS:
        try:
            scaled_px, scaled_py = rescale(norm_px, norm_py, cw, ch)
            labels = place_labels_fn(cw, ch, scaled_px, scaled_py, LABEL_W, LABEL_H)
        except Exception:
            continue

        if len(labels) != len(norm_px):
            continue

        for i in range(len(norm_px)):
            expected_x = baseline_labels[i][0] * cw / BASELINE_SCENARIO[0]
            expected_y = baseline_labels[i][1] * ch / BASELINE_SCENARIO[1]
            dx = (labels[i][0] - expected_x) / cw
            dy = (labels[i][1] - expected_y) / ch
            total_disp += math.sqrt(dx * dx + dy * dy)
            count += 1

    if count == 0:
        return 0.0

    avg_disp = total_disp / count
    return 1.0 - math.tanh(avg_disp / 0.1)


def score_single_config(place_labels_fn, norm_px, norm_py, cw, ch):
    """Score a single configuration on a specific canvas size.

    Rescales normalized data to pixel coordinates for the given canvas.
    Returns dict of metric values or None on error.
    """
    px, py = rescale(norm_px, norm_py, cw, ch)
    labels = place_labels_fn(cw, ch, px, py, LABEL_W, LABEL_H)

    if len(labels) != len(norm_px):
        return None

    return {
        "line_overlap": score_line_overlaps(labels, px, py),
        "point_overlap": score_point_overlaps(labels, px, py),
        "label_overlap": score_label_overlaps(labels),
        "clipping": score_clipping(labels, cw, ch),
        "proximity": score_proximity(labels, px, py, cw, ch),
        "stability": score_stability(place_labels_fn, norm_px, norm_py, cw, ch),
        # Detailed per-label breakdowns
        "per_label_line_overlaps": get_per_label_line_overlaps(labels, px, py),
        "per_label_point_overlaps": get_per_label_point_overlaps(labels, px, py),
        "per_label_label_overlaps": get_per_label_label_overlaps(labels),
        "per_label_clipping": get_per_label_clipping(labels, cw, ch),
        "per_label_proximity": get_per_label_proximity(labels, px, py, cw, ch),
    }


# ---------------------------------------------------------------------------
# Main evaluation: average across multiple configs
# ---------------------------------------------------------------------------

def evaluate():
    """Run the full evaluation across multiple configurations and return the score dict."""
    from experiment.placer import place_labels

    configs = generate_all_configs()
    n_configs = len(configs)

    # Accumulators for averaged metrics
    metrics = {
        "line_overlap": 0.0,
        "point_overlap": 0.0,
        "label_overlap": 0.0,
        "clipping": 0.0,
        "proximity": 0.0,
        "stability": 0.0,
    }

    valid_configs = 0
    config_details = []
    for ci, (norm_px, norm_py) in enumerate(configs):
        cw, ch = BASELINE_SCENARIO
        result = score_single_config(place_labels, norm_px, norm_py, cw, ch)
        if result is None:
            return (
                {
                    "score": 999.0,
                    "description": f"Config {ci}: wrong number of labels",
                    "metrics": {"error": "label count mismatch"},
                },
                [],
            )

        # Aggregate averaged metrics
        for k in metrics:
            metrics[k] += result[k]
        valid_configs += 1

        # Collect per-label details for this config
        config_details.append({
            "config": ci,
            "y_direction": "increasing" if norm_py[0] < norm_py[-1] else "decreasing",
            "line_overlaps": result["per_label_line_overlaps"],
            "point_overlaps": result["per_label_point_overlaps"],
            "label_overlaps": result["per_label_label_overlaps"],
            "clipping": result["per_label_clipping"],
            "proximity": result["per_label_proximity"],
        })

    # Average across configs
    if valid_configs == 0:
        return ({"score": 9999.0, "description": "No valid configurations", "metrics": {}}, [])

    for k in metrics:
        metrics[k] /= valid_configs

    # Combined score
    score = (
        10.0 * metrics["line_overlap"]
        + 10.0 * metrics["point_overlap"]
        + 5.0 * metrics["label_overlap"]
        + 20.0 * metrics["clipping"]
        - 0.5 * metrics["proximity"]
        - 0.3 * metrics["stability"]
    )

    description = (
        f"configs={n_configs} "
        f"line_overlap={metrics['line_overlap']:.3f} "
        f"point_overlap={metrics['point_overlap']:.3f} "
        f"label_overlap={metrics['label_overlap']:.3f} "
        f"clipping={metrics['clipping']:.3f} "
        f"proximity={metrics['proximity']:.3f} "
        f"stability={metrics['stability']:.3f}"
    )

    return (
        {
            "score": round(score, 6),
            "description": description,
            "metrics": {k: round(v, 6) for k, v in metrics.items()},
        },
        config_details,
    )


def print_detailed_feedback(result, config_details):
    """Print detailed per-label metrics for LLM feedback."""
    print(f"\nScore: {result['score']:.4f} | {result['description']}", file=sys.stderr)

    total_issues = 0
    for cd in config_details:
        config = cd["config"]
        y_dir = cd["y_direction"]
        issues = []

        for entry in cd["line_overlaps"]:
            if entry["count"] > 0:
                issues.append(f"L{entry['label']}x{entry['count']}")

        for entry in cd["point_overlaps"]:
            if entry["overlaps_point"]:
                issues.append(f"P{entry['label']}({entry['dist_to_point']}px)")

        for entry in cd["label_overlaps"]:
            issues.append(f"{entry['label_i']}x{entry['label_j']}({entry['overlap_frac']:.0%})")

        for entry in cd["clipping"]:
            if entry["clipped"]:
                issues.append(f"C{entry['label']}({entry['visible_frac']:.0%})")

        for entry in cd["proximity"]:
            if entry["dist_px"] > 50:
                issues.append(f"F{entry['label']}({entry['dist_px']:.0f}px)")

        if issues:
            tag = ", ".join(issues)
            print(f"  Config {config} ({y_dir}): {tag}", file=sys.stderr)
            total_issues += len(issues)

    print(f"\nTotal issues: {total_issues}", file=sys.stderr)


if __name__ == "__main__":
    result, config_details = evaluate()
    print_detailed_feedback(result, config_details)
    output = {"score": result["score"], "description": result["description"]}
    print(json.dumps(output))
    sys.exit(0)
